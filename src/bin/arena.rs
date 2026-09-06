use std::path::PathBuf;
use std::sync::Arc;
use wisdom::ab_tt::TranspositionTableAB;
use wisdom::board::{Board, Color, HistoryEntry, RepetitionResult};
use wisdom::nnue::NNUEModel;
use wisdom::search::search_best_move;
use wisdom::ucci::apply_move_with_history;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let num_games: usize = if args.len() > 1 {
        args[1].parse().unwrap_or(20)
    } else {
        20
    };
    let depth_red: usize = if args.len() > 2 {
        args[2].parse().unwrap_or(8)
    } else {
        8
    };
    let depth_black: usize = if args.len() > 3 {
        args[3].parse().unwrap_or(8)
    } else {
        8
    };

    println!("==================================================");
    println!("⚔️ WISDOM NNUE ARENA");
    println!("Số ván đấu: {}", num_games);
    println!("Đỏ: Depth {}", depth_red);
    println!("Đen: Depth {}", depth_black);
    println!("==================================================");

    let model_path = PathBuf::from("wisdom_models/wisdom_nnue.bin");
    let model = match NNUEModel::load_from_file(&model_path) {
        Ok(m) => Arc::new(m),
        Err(e) => {
            eprintln!("Lỗi tải model: {}", e);
            return;
        }
    };

    let mut red_wins = 0;
    let mut black_wins = 0;
    let mut draws = 0;

    let mut tt_red = TranspositionTableAB::new(32);
    let mut tt_black = TranspositionTableAB::new(32);

    for game_idx in 1..=num_games {
        let mut board = Board::new();
        board.set_initial_position();
        let mut history: Vec<HistoryEntry> = Vec::new();
        let mut ply = 0;
        let max_plies = 200;

        let game_result;

        loop {
            if ply >= max_plies {
                game_result = "Hòa (Hết 200 plies)";
                draws += 1;
                break;
            }

            let rep = board.judge_repetition(&history, history.len(), 3);
            match rep {
                RepetitionResult::Loss => {
                    if board.side_to_move == Color::Red {
                        game_result = "Đen thắng (Đỏ vi phạm luật lặp nước cấm)";
                        black_wins += 1;
                    } else {
                        game_result = "Đỏ thắng (Đen vi phạm luật lặp nước cấm)";
                        red_wins += 1;
                    }
                    break;
                }
                RepetitionResult::Win => {
                    if board.side_to_move == Color::Red {
                        game_result = "Đỏ thắng (Đen phạm luật lặp)";
                        red_wins += 1;
                    } else {
                        game_result = "Đen thắng (Đỏ phạm luật lặp)";
                        black_wins += 1;
                    }
                    break;
                }
                RepetitionResult::Draw => {
                    game_result = "Hòa (Chiếu lặp/Đuổi nhau hợp lệ)";
                    draws += 1;
                    break;
                }
                RepetitionResult::Undecided => {}
            }

            let depth = if board.side_to_move == Color::Red {
                depth_red
            } else {
                depth_black
            };

            let tt = if board.side_to_move == Color::Red {
                &mut tt_red
            } else {
                &mut tt_black
            };

            let res = search_best_move(&mut board, &history, depth, None, &model, tt, None, false);

            let best_move = match res.best_move {
                Some(m) => m,
                None => {
                    // Không còn nước đi hợp lệ => Bị chiếu bí hoặc hết nước đi (thua)
                    if board.side_to_move == Color::Red {
                        game_result = "Đen thắng (Đỏ hết nước đi/chiếu bí)";
                        black_wins += 1;
                    } else {
                        game_result = "Đỏ thắng (Đen hết nước đi/chiếu bí)";
                        red_wins += 1;
                    }
                    break;
                }
            };

            let entry = apply_move_with_history(&mut board, best_move);
            history.push(entry);
            ply += 1;
        }

        println!(
            "Ván {:02}/{:02} ({} plies): {} | Tỉ số: Đỏ {} - {} Đen ({} Hòa)",
            game_idx, num_games, ply, game_result, red_wins, black_wins, draws
        );
    }

    println!("\n🏆 KẾT QUẢ CHUNG CUỘC:");
    println!("Đỏ thắng: {} ({:.1}%)", red_wins, red_wins as f32 / num_games as f32 * 100.0);
    println!("Đen thắng: {} ({:.1}%)", black_wins, black_wins as f32 / num_games as f32 * 100.0);
    println!("Hòa: {} ({:.1}%)", draws, draws as f32 / num_games as f32 * 100.0);
}
