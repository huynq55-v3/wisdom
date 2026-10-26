use std::path::PathBuf;
use std::sync::Arc;
use wisdom::ab_tt::TranspositionTableAB;
use wisdom::board::{Board, Color, HistoryEntry, RepetitionResult};
use wisdom::nnue::NNUEModel;
use wisdom::search::search_best_move;
use wisdom::ucci::apply_move_with_history;

struct ArenaConfig {
    model1_path: String,
    model2_path: String,
    model1_name: String,
    model2_name: String,
    num_games: usize,
    depth1: usize,
    depth2: usize,
    max_plies: usize,
    book_path: Option<String>,
    no_book: bool,
}

fn load_arena_model(path_or_keyword: &str) -> Result<NNUEModel, String> {
    if path_or_keyword.eq_ignore_ascii_case("material") || path_or_keyword.eq_ignore_ascii_case("bootstrap") {
        Ok(NNUEModel::new_material_only())
    } else {
        NNUEModel::load_from_file(PathBuf::from(path_or_keyword))
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

fn parse_arena_args() -> ArenaConfig {
    let args: Vec<String> = std::env::args().collect();
    let mut model1_path = "wisdom_models/wisdom_nnue.bin".to_string();
    let mut model2_path = "material".to_string();
    let mut num_games = 20;
    let mut depth1 = 7;
    let mut depth2 = 7;
    let mut max_plies = 500;
    let mut book_path = None;
    let mut no_book = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--m1" | "--model1" => {
                if i + 1 < args.len() {
                    model1_path = args[i + 1].clone();
                    i += 1;
                }
            }
            "--m2" | "--model2" => {
                if i + 1 < args.len() {
                    model2_path = args[i + 1].clone();
                    i += 1;
                }
            }
            "--games" | "-g" => {
                if i + 1 < args.len() {
                    num_games = args[i + 1].parse().unwrap_or(num_games);
                    i += 1;
                }
            }
            "--depth" | "-d" => {
                if i + 1 < args.len() {
                    let d = args[i + 1].parse().unwrap_or(depth1);
                    depth1 = d;
                    depth2 = d;
                    i += 1;
                }
            }
            "--depth1" => {
                if i + 1 < args.len() {
                    depth1 = args[i + 1].parse().unwrap_or(depth1);
                    i += 1;
                }
            }
            "--depth2" => {
                if i + 1 < args.len() {
                    depth2 = args[i + 1].parse().unwrap_or(depth2);
                    i += 1;
                }
            }
            "--max-plies" => {
                if i + 1 < args.len() {
                    max_plies = args[i + 1].parse().unwrap_or(max_plies);
                    i += 1;
                }
            }
            "--book" => {
                if i + 1 < args.len() {
                    book_path = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--no-book" => {
                no_book = true;
            }
            _ => {
                if let Ok(g) = args[i].parse::<usize>() {
                    num_games = g;
                }
            }
        }
        i += 1;
    }

    let model1_name = if model1_path.eq_ignore_ascii_case("material") {
        "Material Baseline".to_string()
    } else {
        std::path::Path::new(&model1_path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(&model1_path)
            .to_string()
    };

    let model2_name = if model2_path.eq_ignore_ascii_case("material") {
        "Material Baseline".to_string()
    } else {
        std::path::Path::new(&model2_path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(&model2_path)
            .to_string()
    };

    ArenaConfig {
        model1_path,
        model2_path,
        model1_name,
        model2_name,
        num_games,
        depth1,
        depth2,
        max_plies,
        book_path,
        no_book,
    }
}

fn main() {
    let config = parse_arena_args();

    println!("==================================================");
    println!("⚔️ WISDOM ARENA: ĐỐI ĐẦU 2 MÔ HÌNH (MODEL A vs MODEL B)");
    println!("🔵 Model 1 (Challenger): {} (Depth {})", config.model1_name, config.depth1);
    println!("🔴 Model 2 (Defender):   {} (Depth {})", config.model2_name, config.depth2);
    println!("🎮 Tổng số ván đấu: {}", config.num_games);
    println!("🔄 Luật đổi bên: Ván lẻ Model 1 đi Tiên (Đỏ), ván chẵn Model 2 đi Tiên (Đỏ)");

    let opening_fens = if !config.no_book {
        load_opening_fens(config.book_path.as_deref())
    } else {
        println!("⚠️ Không dùng sách khai cuộc (--no-book): Đấu từ thế ban đầu mặc định.");
        Vec::new()
    };

    println!("==================================================");

    let m1 = match load_arena_model(&config.model1_path) {
        Ok(m) => Arc::new(m),
        Err(e) => {
            eprintln!("❌ Lỗi tải Model 1 từ '{}': {}", config.model1_path, e);
            return;
        }
    };

    let m2 = match load_arena_model(&config.model2_path) {
        Ok(m) => Arc::new(m),
        Err(e) => {
            eprintln!("❌ Lỗi tải Model 2 từ '{}': {}", config.model2_path, e);
            return;
        }
    };

    let mut m1_wins = 0;
    let mut m2_wins = 0;
    let mut draws = 0;

    for game_idx in 1..=config.num_games {
        // Đổi bên luân phiên: Ván lẻ M1 cầm Đỏ, ván chẵn M2 cầm Đỏ
        let m1_is_red = (game_idx % 2) == 1;

        let (red_name, red_model, red_depth) = if m1_is_red {
            (&config.model1_name, Arc::clone(&m1), config.depth1)
        } else {
            (&config.model2_name, Arc::clone(&m2), config.depth2)
        };

        let (black_name, black_model, black_depth) = if m1_is_red {
            (&config.model2_name, Arc::clone(&m2), config.depth2)
        } else {
            (&config.model1_name, Arc::clone(&m1), config.depth1)
        };

        let mut board = Board::new();
        let fen_info = if !opening_fens.is_empty() {
            let fen_idx = ((game_idx - 1) / 2) % opening_fens.len();
            match Board::from_fen(&opening_fens[fen_idx]) {
                Ok(b) => {
                    board = b;
                    format!(" | Khai cuộc #{:02}", fen_idx + 1)
                }
                Err(_) => {
                    board.set_initial_position();
                    String::new()
                }
            }
        } else {
            board.set_initial_position();
            String::new()
        };

        let mut history: Vec<HistoryEntry> = Vec::new();
        let mut ply = 0;

        let mut tt_red = TranspositionTableAB::new(32);
        let mut tt_black = TranspositionTableAB::new(32);

        let game_result: String;
        let winner_is_red: Option<bool>; // Some(true) = Red, Some(false) = Black, None = Draw

        loop {
            if ply >= config.max_plies {
                game_result = format!("Hòa (Đạt giới hạn {} plies)", config.max_plies);
                winner_is_red = None;
                break;
            }

            let rep = board.judge_repetition(&history, history.len(), 3);
            match rep {
                RepetitionResult::Loss => {
                    if board.side_to_move == Color::Red {
                        game_result = format!("{} (Đen) thắng do Đỏ phạm luật lặp cấm", black_name);
                        winner_is_red = Some(false);
                    } else {
                        game_result = format!("{} (Đỏ) thắng do Đen phạm luật lặp cấm", red_name);
                        winner_is_red = Some(true);
                    }
                    break;
                }
                RepetitionResult::Win => {
                    if board.side_to_move == Color::Red {
                        game_result = format!("{} (Đỏ) thắng do Đen phạm luật lặp cấm", red_name);
                        winner_is_red = Some(true);
                    } else {
                        game_result = format!("{} (Đen) thắng do Đỏ phạm luật lặp cấm", black_name);
                        winner_is_red = Some(false);
                    }
                    break;
                }
                RepetitionResult::Draw => {
                    game_result = "Hòa (Chiếu/Đuổi nhau hợp lệ)".to_string();
                    winner_is_red = None;
                    break;
                }
                RepetitionResult::Undecided => {}
            }

            let (current_model, current_depth, current_tt) = if board.side_to_move == Color::Red {
                (&red_model, red_depth, &mut tt_red)
            } else {
                (&black_model, black_depth, &mut tt_black)
            };

            let res = search_best_move(&mut board, &history, current_depth, None, current_model, current_tt, None, false);

            let best_move = match res.best_move {
                Some(m) => m,
                None => {
                    if board.side_to_move == Color::Red {
                        game_result = format!("{} (Đen) thắng (Đỏ bị chiếu bí/hết nước)", black_name);
                        winner_is_red = Some(false);
                    } else {
                        game_result = format!("{} (Đỏ) thắng (Đen bị chiếu bí/hết nước)", red_name);
                        winner_is_red = Some(true);
                    }
                    break;
                }
            };

            let entry = apply_move_with_history(&mut board, best_move);
            history.push(entry);
            ply += 1;
        }

        match winner_is_red {
            Some(true) => {
                if m1_is_red {
                    m1_wins += 1;
                } else {
                    m2_wins += 1;
                }
            }
            Some(false) => {
                if m1_is_red {
                    m2_wins += 1;
                } else {
                    m1_wins += 1;
                }
            }
            None => {
                draws += 1;
            }
        }

        println!(
            "Ván {:02}/{:02} ({} plies) [Đỏ: {} | Đen: {}{}] -> {} | Tỉ số: {} {} - {} {} ({} Hòa)",
            game_idx, config.num_games, ply, red_name, black_name, fen_info, game_result,
            config.model1_name, m1_wins, m2_wins, config.model2_name, draws
        );
    }

    let score1 = m1_wins as f64 + 0.5 * draws as f64;
    let score2 = m2_wins as f64 + 0.5 * draws as f64;
    let winrate1 = (score1 / config.num_games as f64) * 100.0;
    let winrate2 = (score2 / config.num_games as f64) * 100.0;

    // Tính ước lượng chênh lệch Elo (Elo diff)
    let elo_diff = if winrate1 >= 99.9 {
        "+800 (Áp đảo hoàn toàn)".to_string()
    } else if winrate1 <= 0.1 {
        "-800 (Thua hoàn toàn)".to_string()
    } else {
        let p = score1 / config.num_games as f64;
        let diff = -400.0 * (1.0 / p - 1.0).log10();
        format!("{:+0.1}", diff)
    };

    println!("\n==================================================");
    println!("🏆 KẾT QUẢ ARENA CHUNG CUỘC:");
    println!("🔵 {}: {} thắng, {} hòa, {} thua | Điểm: {:.1}/{} ({:.1}%)",
        config.model1_name, m1_wins, draws, m2_wins, score1, config.num_games, winrate1
    );
    println!("🔴 {}: {} thắng, {} hòa, {} thua | Điểm: {:.1}/{} ({:.1}%)",
        config.model2_name, m2_wins, draws, m1_wins, score2, config.num_games, winrate2
    );
    println!("📈 Ước lượng Elo Diff: {} Elo (Model 1 so với Model 2)", elo_diff);

    if winrate1 >= 55.0 {
        println!("✨ KẾT LUẬN: Model 1 ĐẠT CHUẨN THĂNG HẠNG (PROMOTION ACCEPTED) 🎉");
    } else if winrate1 >= 50.0 {
        println!("⚖️ KẾT LUẬN: Hai model tương đương trình độ (Marginal / Keep Testing)");
    } else {
        println!("⚠️ KẾT LUẬN: Model 1 THUA Model 2 (PROMOTION REJECTED)");
    }
    println!("==================================================");
}
