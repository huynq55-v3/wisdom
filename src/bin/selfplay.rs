use std::fs::OpenOptions;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use rand::rng;
use rand::seq::IndexedRandom;
use rand::RngExt;
use wisdom::ab_tt::{TranspositionTableAB, VALUE_MATE_THRESHOLD};
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
    bootstrap: bool,
    overwrite: bool,
    book_path: Option<String>,
    no_book: bool,
    epsilon: f32,
}

impl Default for SelfplayConfig {
    fn default() -> Self {
        Self {
            num_games: 50,
            num_threads: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4),
            search_depth: 7,
            random_top_n: 3, // Mặc định khai cuộc
            max_game_plies: 500,
            max_buffer_size: 3_000_000, // 3 triệu positions (~288 MB) chuẩn FIFO
            output_path: PathBuf::from("wisdom_data/replay_buffer.bin"),
            model_path: PathBuf::from("wisdom_models/wisdom_nnue.bin"),
            bootstrap: false,
            overwrite: false,
            book_path: None,
            no_book: false,
            epsilon: 0.1, // 10% cơ hội chọn nước đi ngẫu nhiên hoàn toàn để khám phá
        }
    }
}

fn load_opening_fens(path: Option<&str>) -> Vec<String> {
    let mut candidates = Vec::new();
    if let Some(p) = path {
        candidates.push(p.to_string());
    }
    candidates.push("startFEN.txt".to_string());
    candidates.push("./startFEN.txt".to_string());
    candidates.push("wisdom/startFEN.txt".to_string());
    candidates.push("../wisdom/startFEN.txt".to_string());
    candidates.push("/home/huy/Documents/wisdom/startFEN.txt".to_string());

    for candidate in &candidates {
        if let Ok(content) = std::fs::read_to_string(candidate) {
            let fens: Vec<String> = content
                .lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .collect();
            if !fens.is_empty() {
                println!("📖 Đã nạp thành công {} thế cờ khai cuộc từ: {}", fens.len(), candidate);
                return fens;
            }
        }
    }
    Vec::new()
}

fn parse_args() -> SelfplayConfig {
    let mut config = SelfplayConfig::default();
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--bootstrap" | "--material-bootstrap" => {
                config.bootstrap = true;
            }
            "--overwrite" | "--reset" | "--clear" => {
                config.overwrite = true;
            }
            "--book" => {
                if i + 1 < args.len() {
                    config.book_path = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--no-book" => {
                config.no_book = true;
            }
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
            "--max-plies" | "--plies" => {
                if i + 1 < args.len() {
                    config.max_game_plies = args[i + 1].parse().unwrap_or(config.max_game_plies);
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
            "--epsilon" | "-e" => {
                if i + 1 < args.len() {
                    config.epsilon = args[i + 1].parse().unwrap_or(config.epsilon);
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

struct GameResult {
    records: Vec<WisdomBinRecord>,
    reason: &'static str,
    ply: usize,
    final_fen: String,
    violation_detail: Option<String>,
}

fn play_one_game(
    model: &NNUEModel,
    tt: &mut TranspositionTableAB,
    depth: usize,
    opening_fens: &[String],
    max_plies: usize,
    epsilon: f32,
) -> GameResult {
    let mut local_rng = rng();

    let mut use_book_fen = false;
    let mut board = if !opening_fens.is_empty() && local_rng.random_range(0..10) < 7 {
        if let Some(fen) = opening_fens.choose(&mut local_rng) {
            match Board::from_fen(fen) {
                Ok(b) => {
                    use_book_fen = true;
                    b
                }
                Err(_) => {
                    let mut b = Board::new();
                    b.set_initial_position();
                    b
                }
            }
        } else {
            let mut b = Board::new();
            b.set_initial_position();
            b
        }
    } else {
        let mut b = Board::new();
        b.set_initial_position();
        b
    };

    let mut history: Vec<HistoryEntry> = Vec::new();
    let mut pending: Vec<PendingRecord> = Vec::new();
    let mut ply: usize = 0;

    let reason: &'static str;
    let final_outcome_red: i8; // 1 = Red win, 0 = Draw, -1 = Black win
    let mut violation_detail = None;

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
                let is_check = history.len() >= 2 && history[history.len() - 2].is_check;
                let vtype = if is_check {
                    "Trường chiếu (Perpetual Check)"
                } else {
                    "Trường tróc (Perpetual Chase)"
                };
                if board.side_to_move == Color::Red {
                    reason = "Red perpetual rule loss";
                } else {
                    reason = "Black perpetual rule loss";
                }
                violation_detail = Some(format!("{} - {}", reason, vtype));
                final_outcome_red = if board.side_to_move == Color::Red { -1 } else { 1 };
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
                reason = "Draw by repetition";
                final_outcome_red = 0;
                break;
            }
            RepetitionResult::Undecided => {}
        }

        // Phễu khai cuộc (Opening Funnel):
        // - Ply 0..=4: Top 8 nước, lọc chênh lệch score <= 50 cp (bao quát đủ các thế khai cuộc)
        // - Ply 5..=12: Top 3 nước, lọc chênh lệch score <= 40 cp (biến hóa chuyển sang trung cuộc)
        // - Ply >= 13: 100% Top 1 (tối ưu tốc độ, trung tàn cuộc đánh chuẩn xác, không đi ngớ ngẩn)
        let (current_top_n, max_score_drop) = if ply <= 4 {
            (8, 50)
        } else if ply <= 12 {
            (3, 40)
        } else {
            (1, 0)
        };

        let candidates = search_top_moves(&mut board, &history, depth, current_top_n, model, tt);
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

        // Điểm chuẩn của thế cờ LUÔN là điểm của nước đi tốt nhất (candidates[0])
        let best_eval_score = candidates[0].1;

        // Epsilon-greedy: xác suất epsilon (mặc định 10%) chọn một nước đi ngẫu nhiên hoàn toàn trong các nước hợp lệ!
        let is_random_exploration = epsilon > 0.0 && local_rng.random_range(0.0..1.0f32) < epsilon;

        let chosen_move = if is_random_exploration {
            let all_moves = board.generate_all_moves();
            let moving_side = board.side_to_move;
            let mut legal_moves = Vec::with_capacity(all_moves.len());
            for m in all_moves {
                let undo = board.make_move(m);
                if !board.kings_facing() && !board.is_in_check(moving_side) {
                    legal_moves.push(m);
                }
                board.unmake_move(m, undo);
            }
            if let Some(&m) = legal_moves.choose(&mut local_rng) {
                m
            } else {
                candidates[0].0
            }
        } else if current_top_n > 1 && candidates.len() > 1 {
            let valid_candidates: Vec<_> = candidates
                .iter()
                .filter(|&&(_, s)| (best_eval_score - s) <= max_score_drop)
                .collect();
            if valid_candidates.is_empty() {
                candidates[0].0
            } else {
                valid_candidates.choose(&mut local_rng).unwrap().0
            }
        } else {
            candidates[0].0
        };

        // CHỈ LƯU THẾ CỜ TĨNH (Quiet Position) theo chuẩn Stockfish:
        // 1. Không lưu nếu bên đang đi đang bị chiếu (in_check)
        // 2. Không lưu nếu nước đi tốt nhất là nước ăn quân (tránh làm mù giá trị quân cờ ở depth 0)
        // 3. Không lưu nếu là thế cờ sát cục tuyệt đối (tactical mate sequence)
        // 4. Bỏ qua các ply đầu tiên (2 ply nếu lấy từ book FEN, 6 ply nếu bắt đầu từ cờ xuất phát)
        let in_check = board.is_in_check(board.side_to_move);
        let is_capture = candidates[0].0.is_capture() || board.piece_at(candidates[0].0.to_sq()).is_some();
        let is_mate = best_eval_score.abs() >= VALUE_MATE_THRESHOLD;
        let min_ply_to_save = if use_book_fen { 2 } else { 6 };

        if !in_check && !is_capture && !is_mate && ply >= min_ply_to_save {
            pending.push(PendingRecord {
                board: board.clone(),
                score: best_eval_score,
                ply: ply as u16,
                stm: board.side_to_move,
            });
        }

        // Đi nước cờ
        let entry = apply_move_with_history(&mut board, chosen_move);
        history.push(entry);
        ply += 1;
    }

    let final_fen = board.to_fen();

    // LỌC ĐỘC TỐ DỮ LIỆU:
    // Nếu ván cờ kết thúc vì phạm luật lặp (Perpetual Rule Loss),
    // loại bỏ toàn bộ dữ liệu ván đấu này để tránh gán nhãn sai lệch (credit assignment poisoning).
    if reason.contains("perpetual rule loss") {
        return GameResult {
            records: Vec::new(),
            reason,
            ply,
            final_fen,
            violation_detail,
        };
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

    GameResult {
        records,
        reason,
        ply,
        final_fen,
        violation_detail,
    }
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

    let opening_fens = if !config.no_book {
        load_opening_fens(config.book_path.as_deref())
    } else {
        Vec::new()
    };
    let opening_fens = Arc::new(opening_fens);

    println!("===============================================================");
    println!("⚡️ WISDOM SELF-PLAY GAME GENERATOR (ALPHA-BETA + NNUE)");
    println!("🎮 Số ván đấu mục tiêu: {}", config.num_games);
    println!("🧵 Số luồng CPU song song: {}", config.num_threads);
    println!("🔍 Độ sâu tìm kiếm (Depth): {} plies", config.search_depth);
    println!("🌪️ Phễu khai cuộc (Funnel): Ply 0-4 (Top 8, Δ≤50cp) → Ply 5-12 (Top 3, Δ≤40cp) → Ply ≥13 (Top 1)");
    if !opening_fens.is_empty() {
        println!("📖 Sách khai cuộc: {} thế cờ nạp từ startFEN (70% mở từ book, 30% khởi điểm)", opening_fens.len());
    } else {
        println!("📖 Sách khai cuộc: Không dùng (100% từ bàn cờ ban đầu)");
    }
    println!("⏱️ Giới hạn nước đi tối đa: {} plies/ván", config.max_game_plies);
    println!("🎲 Khám phá Epsilon-greedy: {:.1}% cơ hội chọn nước ngẫu nhiên (--epsilon {:.2})", config.epsilon * 100.0, config.epsilon);
    println!("♻️ Replay Buffer FIFO tối đa: {} positions (~{:.1} MB)", config.max_buffer_size, (config.max_buffer_size * RECORD_SIZE) as f32 / (1024.0 * 1024.0));
    println!("📦 File xuất nhị phân: {:?}", config.output_path);
    if config.bootstrap {
        println!("🌟 Chế độ: Material Bootstrap (Vòng 0 - Dùng hàm vật chất board.evaluate())");
    } else {
        println!("🧠 Model NNUE: {:?}", config.model_path);
    }
    println!("===============================================================");

    // Đảm bảo thư mục đầu ra tồn tại
    if let Some(parent) = config.output_path.parent() {
        std::fs::create_dir_all(parent).unwrap_or(());
    }

    // Nạp NNUE model hoặc chạy chế độ Material Bootstrap
    let model = if config.bootstrap {
        println!("🌟 Đang khởi chạy Material Bootstrap Engine (không cần model ban đầu)...");
        Arc::new(NNUEModel::new_material_only())
    } else {
        match NNUEModel::load_from_file(&config.model_path) {
            Ok(m) => Arc::new(m),
            Err(e) => {
                eprintln!("❌ Lỗi: Không thể tải mô hình NNUE từ {:?}: {}", config.model_path, e);
                eprintln!("💡 Gợi ý: Dùng cờ --bootstrap để tự động sinh dữ liệu khởi điểm bằng hàm vật chất mà không cần file model.");
                std::process::exit(1);
            }
        }
    };

    let games_completed = Arc::new(AtomicUsize::new(0));
    let total_positions = Arc::new(AtomicUsize::new(0));
    let red_wins = Arc::new(AtomicUsize::new(0));
    let black_wins = Arc::new(AtomicUsize::new(0));
    let draws = Arc::new(AtomicUsize::new(0));
    let dropped_games = Arc::new(AtomicUsize::new(0));

    // File writer được bảo vệ bởi Mutex (Ghi đè nếu có cờ --overwrite, mặc định là GHI DỒN append)
    let file = if config.overwrite {
        println!("🗑️  Phát hiện cờ --overwrite: Đặt lại (reset) buffer và ghi mới hoàn toàn!");
        OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&config.output_path)
            .expect("Không thể mở file nhị phân đầu ra")
    } else {
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&config.output_path)
            .expect("Không thể mở file nhị phân đầu ra")
    };
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
        let dropped_clone = Arc::clone(&dropped_games);
        let writer_clone = Arc::clone(&file_writer);
        let opening_fens_clone = Arc::clone(&opening_fens);

        let target_games = config.num_games;
        let depth = config.search_depth;
        let max_plies = config.max_game_plies;
        let epsilon = config.epsilon;

        handles.push(thread::spawn(move || {
            let mut tt = TranspositionTableAB::new(16); // 16MB TT per thread

            loop {
                let current_game = games_comp.fetch_add(1, Ordering::SeqCst);
                if current_game >= target_games {
                    break;
                }

                let res = play_one_game(&model_clone, &mut tt, depth, &opening_fens_clone, max_plies, epsilon);

                let is_rule_violation = res.reason.contains("perpetual rule loss");

                if is_rule_violation {
                    dropped_clone.fetch_add(1, Ordering::Relaxed);
                    println!(
                        "\n🚨 [Ván #{:>3}] THỰC SỰ VI PHẠM LUẬT LẶP: {} tại ply {}\n   ↳ FEN kết thúc: {}\n",
                        current_game + 1,
                        res.violation_detail.as_deref().unwrap_or(res.reason),
                        res.ply,
                        res.final_fen
                    );
                } else {
                    // Ghi nhận bên thắng/thua chính xác (kể cả khi ván thắng sát cục quá sớm không có thế cờ tĩnh để lưu)
                    if res.reason.contains("Red checkmated") || res.reason.contains("Black won") {
                        b_wins.fetch_add(1, Ordering::Relaxed);
                    } else if res.reason.contains("Black checkmated") || res.reason.contains("Red won") {
                        r_wins.fetch_add(1, Ordering::Relaxed);
                    } else if let Some(last_rec) = res.records.last() {
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
                    } else {
                        d_draws.fetch_add(1, Ordering::Relaxed);
                    }

                    // Chỉ ghi dữ liệu vào file nếu có thế cờ tĩnh hợp lệ
                    if !res.records.is_empty() {
                        let num_pos = res.records.len();
                        total_pos.fetch_add(num_pos, Ordering::Relaxed);

                        // Ghi vào file nhị phân
                        let mut writer = writer_clone.lock().unwrap();
                        for rec in &res.records {
                            writer.write_all(rec.as_bytes()).unwrap();
                        }
                    }
                }

                let finished = current_game + 1;
                let elapsed = start_time.elapsed().as_secs_f32();
                let gps = finished as f32 / elapsed.max(0.001);
                let pos_count = total_pos.load(Ordering::Relaxed);
                let pps = pos_count as f32 / elapsed.max(0.001);
                let dropped = dropped_clone.load(Ordering::Relaxed);

                if finished % 5 == 0 || finished == target_games {
                    println!(
                        "[{:>3}/{:>3}] {:5.1}s | {:.1} games/s | {:6.0} pos/s | Tổng positions: {:6} | Đỏ: {} Đen: {} Hòa: {} | Hủy (vi phạm): {}",
                        finished,
                        target_games,
                        elapsed,
                        gps,
                        pps,
                        pos_count,
                        r_wins.load(Ordering::Relaxed),
                        b_wins.load(Ordering::Relaxed),
                        d_draws.load(Ordering::Relaxed),
                        dropped
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
    let dropped_count = dropped_games.load(Ordering::SeqCst);

    println!("\n🎉 HOÀN TẤT SINH DỮ LIỆU TỰ CHƠI TRONG {:.2} GIÂY!", elapsed);
    println!("📊 Số vị trí sinh mới hợp lệ: {} positions", session_pos);
    println!(
        "🏆 Tỉ lệ kết quả: Đỏ thắng {} | Đen thắng {} | Hòa {} (Đã loại bỏ {} ván vi phạm)",
        red_wins.load(Ordering::SeqCst),
        black_wins.load(Ordering::SeqCst),
        draws.load(Ordering::SeqCst),
        dropped_count
    );

    // Áp dụng cơ chế FIFO Replay Buffer
    if let Err(e) = enforce_fifo_buffer_limit(&config.output_path, config.max_buffer_size) {
        eprintln!("⚠️ Cảnh báo: Lỗi khi xử lý FIFO Replay Buffer: {}", e);
    }
}
