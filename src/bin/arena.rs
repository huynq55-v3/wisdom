use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crossbeam_channel::Sender;

use wisdom::board::{Board, Color, HistoryEntry, PieceType, RepetitionResult};
use wisdom::eval_queue::{EvalQueue, EvalRequest};
use wisdom::mcts::MCTS;
use wisdom::nn::XiangqiOnnx; // 🎯 SỬA: Đã chuyển sang dùng ONNX Runtime
use wisdom::r#move::Move;
use wisdom::tt::TranspositionTable;
use wisdom::ucci::move_to_ucci_string;

#[derive(Clone, Copy, PartialEq)]
enum GameResult {
    RedWin,
    BlackWin,
    Draw,
}

fn result_to_string(res: &GameResult) -> &'static str {
    match res {
        GameResult::RedWin => "1-0",
        GameResult::BlackWin => "0-1",
        GameResult::Draw => "1/2-1/2",
    }
}

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

// ==========================================================
// HÀM CHƠI 1 VÁN CỜ (ARENA)
// ==========================================================
fn play_arena_game(
    start_fen: &str,
    red_tx: &Sender<EvalRequest>,
    black_tx: &Sender<EvalRequest>,
    sims: usize,
) -> (GameResult, String) {
    let mut board = Board::from_fen(start_fen).expect("Lỗi đọc FEN khởi tạo");
    let mut history = Vec::new();
    let mut pgn_moves = String::new();

    // 🎯 2 CÂY MCTS RIÊNG BIỆT CHO 2 MODEL (Ngăn chặn đọc trộm suy nghĩ)
    let mcts_red = MCTS::new(500_000);   // Đủ cho 800 sims
    let mcts_black = MCTS::new(500_000); 

    // 🎯 2 BẢNG TT (HASH) RIÊNG BIỆT (Size: 1MB mỗi bảng)
    let tt_red = TranspositionTable::new(2);
    let tt_black = TranspositionTable::new(2);

    let mut move_count = 0;
    // PGN move number đếm tiếp từ vị trí FEN (thường là nước 5 hoặc 6)
    let mut move_number = 1; 

    loop {
        let legal_moves = get_all_legal_moves(&mut board);
        if legal_moves.is_empty() {
            if board.is_in_check(board.side_to_move) {
                return (
                    if board.side_to_move == Color::Red { GameResult::BlackWin } else { GameResult::RedWin },
                    pgn_moves
                );
            } else {
                return (GameResult::Draw, pgn_moves);
            }
        }

        if move_count > 400 {
            return (GameResult::Draw, pgn_moves);
        }

        let rep = board.judge_repetition(&history, history.len(), 1);
        match rep {
            RepetitionResult::Draw => return (GameResult::Draw, pgn_moves),
            RepetitionResult::Loss => return (
                if board.side_to_move == Color::Red { GameResult::BlackWin } else { GameResult::RedWin },
                pgn_moves
            ),
            RepetitionResult::Win => return (
                if board.side_to_move == Color::Red { GameResult::RedWin } else { GameResult::BlackWin },
                pgn_moves
            ),
            _ => {}
        }

        // 🎯 LỰA CHỌN MCTS & TT THEO LƯỢT ĐI
        let (best_move, _) = if board.side_to_move == Color::Red {
            // Lượt Đỏ: Dùng Não Đỏ (MCTS Đỏ + Queue Đỏ + Hash Đỏ)
            mcts_red.search_best_move(&board, &history, sims, red_tx, &tt_red, 1, false)
        } else {
            // Lượt Đen: Dùng Não Đen (MCTS Đen + Queue Đen + Hash Đen)
            mcts_black.search_best_move(&board, &history, sims, black_tx, &tt_black, 1, false)
        };

        if board.side_to_move == Color::Red {
            pgn_moves.push_str(&format!("{}. {} ", move_number, move_to_ucci_string(best_move)));
            move_number += 1;
        } else {
            pgn_moves.push_str(&format!("{} ", move_to_ucci_string(best_move)));
        }

        // Cập nhật bàn cờ và lịch sử (Giữ nguyên logic của bác)
        let is_capture = board.piece_at(best_move.to_sq()).is_some();
        let piece = board.piece_at(best_move.from_sq()).unwrap();
        let is_reversible = !is_capture && (piece.piece_type != PieceType::Pawn || {
            let (from_row, _) = Board::square_to_coord(best_move.from_sq());
            let (to_row, _) = Board::square_to_coord(best_move.to_sq());
            from_row == to_row
        });

        let current_side = board.side_to_move;
        let pre_threats = if is_reversible { board.get_unprotected_threats(current_side) } else { 0 };

        board.make_move(best_move);
        let gives_check = board.is_in_check(board.side_to_move);
        
        let chased_set = if is_reversible && !gives_check {
            let post_threats = board.get_unprotected_threats(current_side);
            post_threats & !pre_threats
        } else { 0 };

        history.push(HistoryEntry {
            hash: board.zobrist_key,
            is_check: gives_check,
            chased_set,
            is_reversible,
        });

        move_count += 1;
    }
}

// ==========================================================
// HÀM MAIN - QUẢN LÝ ĐA LUỒNG & BATCHING
// ==========================================================
fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Cách dùng: cargo run --bin arena --release <MODEL_V0_PATH> <MODEL_V1_PATH> [PGN_OUT_FILE]");
        std::process::exit(1);
    }

    let model_v0_path = &args[1];
    let model_v1_path = &args[2];
    let pgn_out = if args.len() > 3 { args[3].clone() } else { "arena_results.pgn".to_string() };

    let simulations = 800; // Hoặc 800 tùy ý bác

    // 1. ĐỌC FILE FEN KHAI CUỘC (Đảm bảo file startFEN.txt nằm cùng thư mục chạy lệnh)
    let fen_file_path = "startFEN.txt";
    let fens: Vec<String> = {
        let file = File::open(fen_file_path).expect("Không tìm thấy file startFEN.txt");
        let reader = BufReader::new(file);
        reader.lines().map(|l| l.unwrap().trim().to_string()).filter(|l| !l.is_empty()).collect()
    };
    
    // Yêu cầu: 256 FEN. Nếu thiếu/thừa hệ thống vẫn chạy dựa trên số lượng file thực tế
    let num_fens = fens.len();
    let total_games = num_fens * 2; // Đổi màu quân -> Tổng 512 ván

    println!("=====================================================");
    println!("⚔️  ARENA THI ĐẤU: {} vs {}", model_v0_path, model_v1_path);
    println!("📖 Khai cuộc       : Đã nạp {} FENs từ '{}'", num_fens, fen_file_path);
    println!("🎮 Tổng số ván đấu : {} (Mỗi bên cầm Đỏ 50%)", total_games);
    println!("🧠 Simulations     : {}", simulations);
    println!("=====================================================\n");

    // 2. TẢI 2 MODEL ONNX (Dùng CPU hoặc GPU tùy thuộc cấu hình ONNX Runtime của bác)
    println!("📥 Đang nạp Model V0 (Cũ)...");
    let model_v0 = XiangqiOnnx::new(model_v0_path);
    println!("📥 Đang nạp Model V1 (Mới)...");
    let model_v1 = XiangqiOnnx::new(model_v1_path);

    // 3. KHỞI TẠO 2 ỐNG NƯỚC (EVAL QUEUE) ĐỘC LẬP
    // Batch 128 (hoặc 64) là vừa đủ để GPU nhai mượt mà từ 128 luồng
    let queue_v0 = EvalQueue::new(model_v0, 512, 100); 
    let queue_v1 = EvalQueue::new(model_v1, 512, 100);

    let pgn_file = File::create(&pgn_out).expect("Không tạo được file PGN");
    let pgn_writer = Arc::new(Mutex::new(BufWriter::new(pgn_file)));

    let v0_wins = Arc::new(Mutex::new(0));
    let v1_wins = Arc::new(Mutex::new(0));
    let draws = Arc::new(Mutex::new(0));
    let games_completed = Arc::new(Mutex::new(0));

    let start_time = Instant::now();

    // 4. QUẢN LÝ LUỒNG (THREAD POOLING TỰ CHẾ ĐỂ TRÁNH NGỘP OS)
    // Hệ điều hành không thích 512 luồng chạy CÙNG LÚC. Ta chia thành các mẻ nhỏ (ví dụ 128 luồng chạy đồng thời)
    // Mỗi luồng cày xong ván này sẽ bốc ván khác cày tiếp.
    
    // Tạo danh sách 512 cấu hình trận đấu (FEN, Ai_Cầm_Đỏ)
    let mut match_configs = Vec::with_capacity(total_games);
    for (i, fen) in fens.iter().enumerate() {
        // Ván 1: V1 cầm Đỏ, V0 cầm Đen
        match_configs.push((i * 2, fen.clone(), true));  
        // Ván 2: V0 cầm Đỏ, V1 cầm Đen (Đảo màu)
        match_configs.push((i * 2 + 1, fen.clone(), false)); 
    }

    let shared_configs = Arc::new(Mutex::new(match_configs.into_iter()));
    
    // Tối đa 128 luồng hoạt động song song
    let max_concurrent_threads = 512; 

    std::thread::scope(|s| {
        for _ in 0..max_concurrent_threads {
            let configs_clone = Arc::clone(&shared_configs);
            let tx_v0 = &queue_v0.tx;
            let tx_v1 = &queue_v1.tx;
            
            let pgn_writer_clone = Arc::clone(&pgn_writer);
            let v0_w = Arc::clone(&v0_wins);
            let v1_w = Arc::clone(&v1_wins);
            let dr = Arc::clone(&draws);
            let gc = Arc::clone(&games_completed);

            s.spawn(move || {
                loop {
                    // Bốc 1 ván cờ từ danh sách
                    let config = {
                        let mut it = configs_clone.lock().unwrap();
                        it.next()
                    };

                    match config {
                        Some((game_id, start_fen, is_v1_red)) => {
                            // Cấp phát ống nước đúng chủ nhân
                            let (red_tx, black_tx) = if is_v1_red {
                                (tx_v1, tx_v0)
                            } else {
                                (tx_v0, tx_v1)
                            };

                            let (result, pgn_content) = play_arena_game(&start_fen, red_tx, black_tx, simulations);

                            // --- Ghi PGN ---
                            {
                                let mut f = pgn_writer_clone.lock().unwrap();
                                let red_player = if is_v1_red { "Model V1 (Mới)" } else { "Model V0 (Cũ)" };
                                let black_player = if is_v1_red { "Model V0 (Cũ)" } else { "Model V1 (Mới)" };

                                writeln!(f, "[Event \"Arena v0 vs v1\"]\n[Round \"{}\"]\n[White \"{}\"]\n[Black \"{}\"]\n[Result \"{}\"]\n[FEN \"{}\"]\n{}\n", 
                                    game_id + 1, red_player, black_player, result_to_string(&result), start_fen, pgn_content.trim()).unwrap();
                                f.flush().unwrap();
                            }

                            // --- Cập nhật điểm số ---
                            {
                                let mut current_v0 = v0_w.lock().unwrap();
                                let mut current_v1 = v1_w.lock().unwrap();
                                let mut current_draws = dr.lock().unwrap();
                                let mut current_completed = gc.lock().unwrap();

                                match result {
                                    GameResult::RedWin => if is_v1_red { *current_v1 += 1 } else { *current_v0 += 1 },
                                    GameResult::BlackWin => if is_v1_red { *current_v0 += 1 } else { *current_v1 += 1 },
                                    GameResult::Draw => *current_draws += 1,
                                }
                                *current_completed += 1;
                                
                                println!("🏁 Xong Trận {}/{}. 📊 Tỉ số hiện tại - V1: {} | V0: {} | Hòa: {}", 
                                    *current_completed, total_games, *current_v1, *current_v0, *current_draws);
                            }
                        }
                        None => break, // Hết việc, luồng này tự động thoát
                    }
                }
            });
        }
    });

    let final_v1 = *v1_wins.lock().unwrap();
    let final_v0 = *v0_wins.lock().unwrap();
    let final_draws = *draws.lock().unwrap();
    
    // Tính Winrate cho V1 (Thắng 1 điểm, Hòa 0.5 điểm)
    let v1_score = final_v1 as f32 + (final_draws as f32 / 2.0);
    let winrate = (v1_score / total_games as f32) * 100.0;

    println!("\n=====================================================");
    println!("🏆 KẾT QUẢ CHUNG CUỘC SAU {:.2?}", start_time.elapsed());
    println!("🟢 Model V1 (Mới) thắng : {}", final_v1);
    println!("🔴 Model V0 (Cũ) thắng  : {}", final_v0);
    println!("⚪ Hòa                  : {}", final_draws);
    println!("📈 WINRATE CỦA V1       : {:.2}%", winrate);
    println!("=====================================================");
    
    if winrate > 55.0 {
        println!("🎉 KẾT LUẬN: MODEL V1 ĐÃ VƯỢT TRỘI, ĐỦ ĐIỀU KIỆN LÊN NGÔI!");
    } else {
        println!("⚠️ KẾT LUẬN: Model V1 chưa đủ mạnh (Winrate < 55%), cần cày thêm Data.");
    }
    
    println!("💾 Lịch sử các ván đấu đã được lưu tại {}", pgn_out);
}