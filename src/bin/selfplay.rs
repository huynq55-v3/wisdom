use crossbeam_channel::Sender;
use std::env;
use std::fs;
use std::io::{BufWriter, Write};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use wisdom::board::{Board, Color, HistoryEntry, PieceType};
use wisdom::eval_queue::{EvalQueue, EvalRequest};
use wisdom::mcts::MCTS;
use wisdom::r#move::Move;
use wisdom::nn::XiangqiOnnx;
use wisdom::tt::TranspositionTable;

// ==========================================================
// CẤU TRÚC LƯU TRỮ
// ==========================================================
#[derive(Clone, Debug)]
pub struct SelfPlayItem {
    pub fen: String,
    pub value: f32,
    pub policy: usize,
}

// ==========================================================
// HÀM TRỢ GIÚP TÌM MODEL MỚI NHẤT
// ==========================================================
fn get_latest_model_version(model_dir: &str) -> usize {
    let mut latest_version = 0;

    if let Ok(entries) = fs::read_dir(model_dir) {
        for entry in entries.flatten() {
            let file_name = entry.file_name().into_string().unwrap_or_default();
            // Lọc các file có dạng: wisdom_net_vX.onnx
            if file_name.starts_with("wisdom_net_v") && file_name.ends_with(".onnx") {
                if let Some(v_str) = file_name
                    .strip_prefix("wisdom_net_v")
                    .and_then(|s| s.strip_suffix(".onnx"))
                {
                    if let Ok(v) = v_str.parse::<usize>() {
                        if v > latest_version {
                            latest_version = v;
                        }
                    }
                }
            }
        }
    }
    latest_version
}

// ==========================================================
// LOGIC CHƠI 1 GAME (CHỈ LẤY FEN)
// ==========================================================
fn get_all_legal_moves(board: &mut Board) -> Vec<Move> {
    let mut all_moves = board.generate_captures();
    all_moves.append(&mut board.generate_quiets());
    let mut legal_moves = Vec::new();
    let moving_side = board.side_to_move;

    for m in all_moves {
        let undo = board.make_move(m);
        let legal = !board.kings_facing() && !board.is_in_check(moving_side);
        board.unmake_move(m, undo);
        if legal {
            legal_moves.push(m);
        }
    }
    legal_moves
}

fn play_game(
    eval_tx: &Sender<EvalRequest>,
    shared_tt: &Arc<TranspositionTable>,
) -> Vec<SelfPlayItem> {
    let mut board = Board::new();
    board.set_initial_position();
    let mut history = Vec::new();

    let mut game_records: Vec<(String, f32, Color, usize)> = Vec::new();
    let mut move_count = 0;
    let winner: Option<Color>;

    let simulations = 800;
    let mcts = MCTS::new(500_000);

    loop {
        let legal_moves = get_all_legal_moves(&mut board);
        if legal_moves.is_empty() {
            winner = if board.is_in_check(board.side_to_move) {
                Some(board.side_to_move.opposite())
            } else {
                None
            };
            break;
        }

        if move_count > 400 {
            winner = None;
            break;
        }

        let rep = board.judge_repetition(&history, history.len(), 1);
        match rep {
            wisdom::board::RepetitionResult::Draw => {
                winner = None;
                break;
            }
            wisdom::board::RepetitionResult::Loss => {
                winner = Some(board.side_to_move.opposite());
                break;
            }
            wisdom::board::RepetitionResult::Win => {
                winner = Some(board.side_to_move);
                break;
            }
            _ => {}
        }

        let (best_move, metrics) =
            mcts.search_best_move(&board, &history, simulations, eval_tx, shared_tt, 1, true);

        // In dấu chấm để báo hiệu game đang chạy mượt
        print!(".");
        let _ = std::io::stdout().flush();

        let normalized_score = (metrics.win_pct / 50.0) - 1.0;
        let current_side = board.side_to_move;

        game_records.push((
            board.to_fen(),
            normalized_score,
            current_side,
            Move::move_to_nn_index(best_move, current_side == Color::Black),
        ));

        let chosen_move = best_move;
        let is_capture = board.piece_at(chosen_move.to_sq()).is_some();
        let piece = board.piece_at(chosen_move.from_sq()).unwrap();
        let is_reversible = !is_capture
            && (piece.piece_type != PieceType::Pawn || {
                let (from_row, _) = Board::square_to_coord(chosen_move.from_sq());
                let (to_row, _) = Board::square_to_coord(chosen_move.to_sq());
                from_row == to_row
            });

        let pre_threats = if is_reversible {
            board.get_unprotected_threats(current_side)
        } else {
            0
        };
        board.make_move(chosen_move);
        let gives_check = board.is_in_check(board.side_to_move);
        let chased_set = if is_reversible && !gives_check {
            let post_threats = board.get_unprotected_threats(current_side);
            post_threats & !pre_threats
        } else {
            0
        };

        history.push(HistoryEntry {
            hash: board.zobrist_key,
            is_check: gives_check,
            chased_set,
            is_reversible,
        });

        move_count += 1;
    }

    // 🎯 ÁP DỤNG CÔNG THỨC ALPHA = 0.8
    let alpha = 0.9;
    match winner {
        None => game_records
            .into_iter()
            .map(|(fen, search_val, _, policy)| SelfPlayItem {
                fen,
                value: search_val * alpha, // Game hòa z = 0
                policy,
            })
            .collect(),
        Some(winning_color) => game_records
            .into_iter()
            .map(|(fen, search_val, side, policy)| {
                let z = if side == winning_color { 1.0 } else { -1.0 };
                SelfPlayItem {
                    fen,
                    value: search_val * alpha + z * (1.0 - alpha),
                    policy,
                }
            })
            .collect(),
    }
}

// ==========================================================
// HÀM MAIN - QUẢN LÝ LUỒNG & BATCHING
// ==========================================================
fn main() {
    // 1. Nhận tham số số vòng (rounds) từ Command Line
    let args: Vec<String> = env::args().collect();
    let rounds: usize = if args.len() > 1 {
        args[1]
            .parse()
            .expect("Vui lòng nhập một số nguyên cho số vòng (VD: cargo run --bin selfplay 10)")
    } else {
        1 // Mặc định chạy 1 vòng
    };

    let model_dir = "./wisdom_models";
    fs::create_dir_all(model_dir).expect("Không thể tạo thư mục models");

    // 2. Tự động quét và Load model mới nhất
    let version = get_latest_model_version(model_dir);
    let model_path = format!("{}/wisdom_net_v{}.onnx", model_dir, version);

    println!("============================================================");
    println!("🚀 KHỞI ĐỘNG HỆ THỐNG SELF-PLAY");
    println!("📦 Model đang dùng : {}", model_path);
    println!(
        "🔄 Số vòng         : {} (Tổng {} games)",
        rounds,
        rounds * 512
    );
    println!("⚙️ Batch Size      : 512 | Timeout: 100 µs");
    println!("============================================================\n");

    // Tải mô hình OpenVINO / ONNX
    let onnx_model = XiangqiOnnx::new(&model_path);

    let eval_queue = EvalQueue::new(onnx_model, 512, 100);
    let all_generated_data = Arc::new(Mutex::new(Vec::new()));
    let shared_tt = Arc::new(TranspositionTable::new(16384));

    let start_time = Instant::now();

    for round in 1..=rounds {
        println!("\n▶️ Bắt đầu Vòng {}/{}...", round, rounds);

        // 4. Khởi tạo chính xác 128 luồng bằng std::thread::scope
        std::thread::scope(|s| {
            for _ in 0..512 {
                let tx = &eval_queue.tx;
                let data_clone = Arc::clone(&all_generated_data);
                let tt_clone = Arc::clone(&shared_tt);

                s.spawn(move || {
                    let records = play_game(tx, &tt_clone);
                    data_clone.lock().unwrap().extend(records);
                });
            }
        });

        println!("\n✅ Hoàn thành Vòng {}.", round);
    }

    // 5. Lưu toàn bộ kết quả vào 1 file duy nhất trong thư mục v{version}
    let data_snapshot = all_generated_data.lock().unwrap();
    if !data_snapshot.is_empty() {
        let output_dir = format!("{}/v{}", model_dir, version);
        fs::create_dir_all(&output_dir).unwrap();

        let file_path = format!(
            "{}/selfplay_data_{}.csv",
            output_dir,
            std::time::UNIX_EPOCH.elapsed().unwrap().as_secs()
        );
        let file = fs::File::create(&file_path).expect("Không thể tạo file data");
        let mut writer = BufWriter::new(file);

        for item in data_snapshot.iter() {
            writeln!(writer, "{},{},{}", item.fen, item.value, item.policy).unwrap();
        }

        println!(
            "\n🎉 XONG! Đã sinh thành công {} FENs.",
            data_snapshot.len()
        );
        println!("💾 Dữ liệu được lưu tại: {}", file_path);
        println!("⏱️ Tổng thời gian chạy: {:.2?}", start_time.elapsed());
    } else {
        println!("\n⚠️ Không có dữ liệu nào được sinh ra.");
    }
}
