use std::fs::OpenOptions;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use rand::rng;
use rand::seq::IndexedRandom;
use wisdom::ab_tt::TranspositionTableAB;
use wisdom::bin_format::{WisdomBinRecord, RECORD_SIZE};
use wisdom::board::{Board, Color, HistoryEntry, RepetitionResult};
use wisdom::nnue::NNUEModel;
use wisdom::search::search_top_moves;
use wisdom::ucci::apply_move_with_history;

struct SelfplayConfig {
    num_games: usize,
    num_threads: usize,
    search_depth: usize,
    random_top_n: usize,
    max_game_plies: usize,
    max_buffer_size: usize,
    output_path: PathBuf,
    model_path: PathBuf,
}

impl Default for SelfplayConfig {
    fn default() -> Self {
        Self {
            num_games: 50,
            num_threads: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4),
            search_depth: 7,
            random_top_n: 3, // Mặc định chọn ngẫu nhiên trong Top 3 nước điểm cao nhất
            max_game_plies: 220,
            max_buffer_size: 1_000_000, // 1 triệu positions (~96 MB) chuẩn FIFO
            output_path: PathBuf::from("wisdom_data/replay_buffer.bin"),
            model_path: PathBuf::from("wisdom_models/wisdom_nnue.bin"),
        }
    }
}

fn parse_args() -> SelfplayConfig {
    let mut config = SelfplayConfig::default();
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--games" | "-g" => {
                if i + 1 < args.len() {
                    config.num_games = args[i + 1].parse().unwrap_or(config.num_games);
                    i += 1;
                }
            }
            "--threads" | "-t" => {
                if i + 1 < args.len() {
                    config.num_threads = args[i + 1].parse().unwrap_or(config.num_threads);
                    i += 1;
                }
            }
            "--depth" | "-d" => {
                if i + 1 < args.len() {
                    config.search_depth = args[i + 1].parse().unwrap_or(config.search_depth);
                    i += 1;
                }
            }
            "--random-plies" | "-r" | "--random-top" | "--top-moves" => {
                if i + 1 < args.len() {
                    config.random_top_n = args[i + 1].parse().unwrap_or(config.random_top_n);
                    i += 1;
                }
            }
            "--buffer-size" | "--max-buffer" => {
                if i + 1 < args.len() {
                    config.max_buffer_size = args[i + 1].parse().unwrap_or(config.max_buffer_size);
                    i += 1;
                }
            }
            "--output" | "-o" => {
                if i + 1 < args.len() {
                    config.output_path = PathBuf::from(&args[i + 1]);
                    i += 1;
                }
            }
            "--model" | "-m" => {
                if i + 1 < args.len() {
                    config.model_path = PathBuf::from(&args[i + 1]);
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }
    config
}

struct PendingRecord {
    board: Board,
    score: i16,
    ply: u16,
    stm: Color,
}

fn play_one_game(
    model: &NNUEModel,
    tt: &mut TranspositionTableAB,
    depth: usize,
    random_top_n: usize,
    max_plies: usize,
) -> (Vec<WisdomBinRecord>, &'static str) {
    let mut board = Board::new();
    board.set_initial_position();
    let mut history: Vec<HistoryEntry> = Vec::new();
    let mut pending: Vec<PendingRecord> = Vec::new();
    let mut ply: usize = 0;

    let mut local_rng = rng();

    let reason: &'static str;
    let final_outcome_red: i8; // 1 = Red win, 0 = Draw, -1 = Black win

    loop {
        if ply >= max_plies {
            reason = "Max plies reached";
            final_outcome_red = 0;
            break;
        }

        // Kiểm tra luật lặp cấm
        let rep = board.judge_repetition(&history, history.len(), 3);
        match rep {
            RepetitionResult::Loss => {
                if board.side_to_move == Color::Red {
                    reason = "Red perpetual rule loss";
                    final_outcome_red = -1;
                } else {
                    reason = "Black perpetual rule loss";
                    final_outcome_red = 1;
                }
                break;
            }
            RepetitionResult::Win => {
                if board.side_to_move == Color::Red {
                    reason = "Red won by opponent rule violation";
                    final_outcome_red = 1;
                } else {
                    reason = "Black won by opponent rule violation";
                    final_outcome_red = -1;
                }
                break;
            }
            RepetitionResult::Draw => {
                reason = "Draw by perpetual check/chase";
                final_outcome_red = 0;
                break;
            }
            RepetitionResult::Undecided => {}
        }

        // Tìm kiếm các nước đi tốt nhất với Alpha-Beta + NNUE
        let candidates = search_top_moves(&mut board, &history, depth, random_top_n, model, tt);
        if candidates.is_empty() {
            // Không còn nước đi hợp lệ => Bên tới lượt bị chiếu bí hoặc hết nước đi (thua)
            if board.side_to_move == Color::Red {
                reason = "Red checkmated / stalemate";
                final_outcome_red = -1;
            } else {
                reason = "Black checkmated / stalemate";
                final_outcome_red = 1;
            }
            break;
        }

        // Chọn ngẫu nhiên trong danh sách Top-N nước có điểm số cao nhất
        let (best_move, eval_score) = *candidates.choose(&mut local_rng).unwrap();

        // Lưu thế cờ trước khi đi
        pending.push(PendingRecord {
            board: board.clone(),
            score: eval_score,
            ply: ply as u16,
            stm: board.side_to_move,
        });

        // Đi nước cờ
        let entry = apply_move_with_history(&mut board, best_move);
        history.push(entry);
        ply += 1;
    }

    // Gán kết quả chung cuộc cho toàn bộ các thế cờ trong ván đấu
    let mut records = Vec::with_capacity(pending.len());
    for p in pending {
        let outcome = match p.stm {
            Color::Red => final_outcome_red,
            Color::Black => -final_outcome_red,
        };
        let rec = WisdomBinRecord::new(&p.board, p.score, outcome, p.ply);
        records.push(rec);
    }

    (records, reason)
}

/// Cơ chế Replay Buffer First In First Out (FIFO):
/// Giữ kích thước buffer luôn ổn định ở mức tối đa max_records (mặc định 1 triệu positions).
/// Cắt bỏ các ván cờ cũ nhất ở đầu file khi buffer vượt ngưỡng.
fn enforce_fifo_buffer_limit(path: &Path, max_records: usize) -> std::io::Result<()> {
    if !path.exists() {
        return Ok(());
    }
    let file_size = std::fs::metadata(path)?.len() as usize;
    let total_records = file_size / RECORD_SIZE;
    if total_records <= max_records {
        return Ok(());
    }

    let records_to_drop = total_records - max_records;
    let drop_bytes = (records_to_drop * RECORD_SIZE) as u64;

    println!(
        "\n♻️ Replay Buffer FIFO: Tổng {} positions vượt giới hạn chuẩn {} positions.",
        total_records, max_records
    );
    println!(
        "   ↳ Cắt bỏ {} positions cũ nhất (FIFO: First-In First-Out)...",
        records_to_drop
    );

    let temp_path = path.with_extension("tmp_fifo");
    {
        let mut reader = std::fs::File::open(path)?;
        reader.seek(SeekFrom::Start(drop_bytes))?;
        let mut writer = std::fs::File::create(&temp_path)?;
        std::io::copy(&mut reader, &mut writer)?;
        writer.flush()?;
    }

    std::fs::rename(&temp_path, path)?;
    let new_size_mb = (max_records * RECORD_SIZE) as f32 / (1024.0 * 1024.0);
    println!(
        "   ✅ Đã đồng bộ Buffer về đúng {} positions ({:.2} MB)!",
        max_records, new_size_mb
    );
    Ok(())
}

fn main() {
    let config = parse_args();

    println!("===============================================================");
    println!("⚡️ WISDOM SELF-PLAY GAME GENERATOR (ALPHA-BETA + NNUE)");
    println!("🎮 Số ván đấu mục tiêu: {}", config.num_games);
    println!("🧵 Số luồng CPU song song: {}", config.num_threads);
    println!("🔍 Độ sâu tìm kiếm (Depth): {} plies", config.search_depth);
    println!("🎲 Chọn nước ngẫu nhiên: Top {} nước điểm số cao nhất (toàn bộ ván)", config.random_top_n);
    println!("♻️ Replay Buffer FIFO tối đa: {} positions (~{:.1} MB)", config.max_buffer_size, (config.max_buffer_size * RECORD_SIZE) as f32 / (1024.0 * 1024.0));
    println!("📦 File xuất nhị phân: {:?}", config.output_path);
    println!("🧠 Model NNUE: {:?}", config.model_path);
    println!("===============================================================");

    // Đảm bảo thư mục đầu ra tồn tại
    if let Some(parent) = config.output_path.parent() {
        std::fs::create_dir_all(parent).unwrap_or(());
    }

    // Nạp NNUE model
    let model = match NNUEModel::load_from_file(&config.model_path) {
        Ok(m) => Arc::new(m),
        Err(e) => {
            eprintln!("❌ Lỗi: Không thể tải mô hình NNUE từ {:?}: {}", config.model_path, e);
            eprintln!("💡 Gợi ý: Hãy kiểm tra file model hoặc chạy script train để tạo model khởi điểm.");
            std::process::exit(1);
        }
    };

    let games_completed = Arc::new(AtomicUsize::new(0));
    let total_positions = Arc::new(AtomicUsize::new(0));
    let red_wins = Arc::new(AtomicUsize::new(0));
    let black_wins = Arc::new(AtomicUsize::new(0));
    let draws = Arc::new(AtomicUsize::new(0));

    // File writer được bảo vệ bởi Mutex
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&config.output_path)
        .expect("Không thể mở file nhị phân đầu ra");
    let file_writer = Arc::new(Mutex::new(BufWriter::with_capacity(1024 * 1024, file)));

    let start_time = Instant::now();
    let mut handles = Vec::new();

    for _thread_id in 0..config.num_threads {
        let model_clone = Arc::clone(&model);
        let games_comp = Arc::clone(&games_completed);
        let total_pos = Arc::clone(&total_positions);
        let r_wins = Arc::clone(&red_wins);
        let b_wins = Arc::clone(&black_wins);
        let d_draws = Arc::clone(&draws);
        let writer_clone = Arc::clone(&file_writer);

        let target_games = config.num_games;
        let depth = config.search_depth;
        let random_top_n = config.random_top_n;
        let max_plies = config.max_game_plies;

        handles.push(thread::spawn(move || {
            let mut tt = TranspositionTableAB::new(16); // 16MB TT per thread

            loop {
                let current_game = games_comp.fetch_add(1, Ordering::SeqCst);
                if current_game >= target_games {
                    break;
                }

                let (records, _reason) = play_one_game(&model_clone, &mut tt, depth, random_top_n, max_plies);

                if let Some(last_rec) = records.last() {
                    // Xác định bên thắng
                    if last_rec.outcome > 0 {
                        if last_rec.side_to_move == 0 {
                            r_wins.fetch_add(1, Ordering::Relaxed);
                        } else {
                            b_wins.fetch_add(1, Ordering::Relaxed);
                        }
                    } else if last_rec.outcome < 0 {
                        if last_rec.side_to_move == 0 {
                            b_wins.fetch_add(1, Ordering::Relaxed);
                        } else {
                            r_wins.fetch_add(1, Ordering::Relaxed);
                        }
                    } else {
                        d_draws.fetch_add(1, Ordering::Relaxed);
                    }
                }

                let num_pos = records.len();
                total_pos.fetch_add(num_pos, Ordering::Relaxed);

                // Ghi vào file nhị phân
                {
                    let mut writer = writer_clone.lock().unwrap();
                    for rec in &records {
                        writer.write_all(rec.as_bytes()).unwrap();
                    }
                }

                let finished = current_game + 1;
                let elapsed = start_time.elapsed().as_secs_f32();
                let gps = finished as f32 / elapsed.max(0.001);
                let pos_count = total_pos.load(Ordering::Relaxed);
                let pps = pos_count as f32 / elapsed.max(0.001);

                if finished % 5 == 0 || finished == target_games {
                    println!(
                        "[{:>3}/{:>3}] {:5.1}s | {:.1} games/s | {:6.0} pos/s | Tổng positions: {:6} | Đỏ: {} Đen: {} Hòa: {}",
                        finished,
                        target_games,
                        elapsed,
                        gps,
                        pps,
                        pos_count,
                        r_wins.load(Ordering::Relaxed),
                        b_wins.load(Ordering::Relaxed),
                        d_draws.load(Ordering::Relaxed)
                    );
                }
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    // Flush & close file writer trước khi kiểm tra FIFO
    {
        let mut writer = file_writer.lock().unwrap();
        writer.flush().unwrap();
    }
    drop(file_writer);

    let elapsed = start_time.elapsed().as_secs_f32();
    let session_pos = total_positions.load(Ordering::SeqCst);

    println!("\n🎉 HOÀN TẤT SINH DỮ LIỆU TỰ CHƠI TRONG {:.2} GIÂY!", elapsed);
    println!("📊 Số vị trí sinh mới trong lượt này: {} positions", session_pos);
    println!(
        "🏆 Tỉ lệ kết quả: Đỏ thắng {} | Đen thắng {} | Hòa {}",
        red_wins.load(Ordering::SeqCst),
        black_wins.load(Ordering::SeqCst),
        draws.load(Ordering::SeqCst)
    );

    // Áp dụng cơ chế FIFO Replay Buffer
    if let Err(e) = enforce_fifo_buffer_limit(&config.output_path, config.max_buffer_size) {
        eprintln!("⚠️ Cảnh báo: Lỗi khi xử lý FIFO Replay Buffer: {}", e);
    }
}
