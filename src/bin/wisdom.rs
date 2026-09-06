use std::io::{self, BufRead};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use wisdom::ab_tt::TranspositionTableAB;
use wisdom::board::{Board, Color, HistoryEntry, PieceType};
use wisdom::nnue::NNUEModel;
use wisdom::r#move::Move;
use wisdom::search::search_best_move;
use wisdom::ucci::{move_to_ucci_string, parse_fen, parse_ucci_move};

fn find_model_path() -> std::path::PathBuf {
    let candidates = [
        std::path::PathBuf::from("./wisdom_models/wisdom_nnue.bin"),
        std::path::PathBuf::from("./wisdom_nnue.bin"),
        std::path::PathBuf::from("/home/huy/Documents/wisdom/wisdom_models/wisdom_nnue.bin"),
    ];

    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let p1 = dir.join("wisdom_models").join("wisdom_nnue.bin");
            if p1.exists() {
                return p1;
            }
            let p2 = dir.join("wisdom_nnue.bin");
            if p2.exists() {
                return p2;
            }
        }
    }

    for p in &candidates {
        if p.exists() {
            return p.clone();
        }
    }

    candidates[0].clone()
}

fn apply_move_with_history(board: &mut Board, m: Move) -> HistoryEntry {
    let current_side = board.side_to_move;
    let is_capture = board.piece_at(m.to_sq()).is_some();
    let piece = board.piece_at(m.from_sq()).unwrap();

    let is_reversible = !is_capture
        && (piece.piece_type != PieceType::Pawn || {
            let (from_row, _) = Board::square_to_coord(m.from_sq());
            let (to_row, _) = Board::square_to_coord(m.to_sq());
            from_row == to_row
        });

    let pre_threats = if is_reversible {
        board.get_unprotected_threats(current_side)
    } else {
        0
    };

    board.make_move(m);
    let gives_check = board.is_in_check(board.side_to_move);
    let chased_set = if is_reversible && !gives_check {
        let post_threats = board.get_unprotected_threats(current_side);
        post_threats & !pre_threats
    } else {
        0
    };

    HistoryEntry {
        hash: board.zobrist_key,
        is_check: gives_check,
        chased_set,
        is_reversible,
    }
}

fn main() {
    let model_path = find_model_path();
    let model = match NNUEModel::load_from_file(&model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("info error Failed to load NNUE model from {:?}: {}", model_path, e);
            return;
        }
    };

    let mut tt = TranspositionTableAB::new(64); // 64 MB TT table
    let mut board = Board::new();
    board.set_initial_position();
    let mut game_history: Vec<HistoryEntry> = Vec::new();
    let stop_signal = Arc::new(AtomicBool::new(false));

    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l.trim().to_string(),
            Err(_) => break,
        };

        if line.is_empty() {
            continue;
        }

        let tokens: Vec<&str> = line.split_whitespace().collect();
        let cmd = tokens[0];

        match cmd {
            "ucci" | "uci" => {
                println!("id name Wisdom NNUE AlphaBeta Engine");
                println!("id author Quang Huy");
                println!("option name Hash type spin default 64 min 1 max 1024");
                if cmd == "uci" {
                    println!("uciok");
                } else {
                    println!("ucciok");
                }
            }
            "isready" => {
                println!("readyok");
            }
            "setoption" => {
                if tokens.len() >= 5 && tokens[1] == "name" && tokens[2].eq_ignore_ascii_case("hash") && tokens[3] == "value" {
                    if let Ok(mb) = tokens[4].parse::<usize>() {
                        tt = TranspositionTableAB::new(mb.clamp(1, 1024));
                        println!("info string Hash set to {} MB", mb);
                    }
                }
            }
            "ucinewgame" => {
                tt.clear();
                game_history.clear();
                board.set_initial_position();
            }
            "position" => {
                if tokens.len() > 1 {
                    let mut moves_idx = tokens.len();
                    if tokens[1] == "startpos" {
                        board.set_initial_position();
                        if let Some(pos) = tokens.iter().position(|&t| t == "moves") {
                            moves_idx = pos;
                        }
                    } else if tokens[1] == "fen" {
                        let moves_token_pos = tokens.iter().position(|&t| t == "moves");
                        let fen_end = moves_token_pos.unwrap_or(tokens.len());
                        let fen = tokens[2..fen_end].join(" ");
                        parse_fen(&mut board, &fen);
                        moves_idx = moves_token_pos.unwrap_or(tokens.len());
                    }

                    game_history.clear();

                    if moves_idx < tokens.len() && tokens[moves_idx] == "moves" {
                        for move_str in &tokens[moves_idx + 1..] {
                            if let Some(m) = parse_ucci_move(&board, move_str) {
                                let entry = apply_move_with_history(&mut board, m);
                                game_history.push(entry);
                            }
                        }
                    }
                }
            }
            "go" => {
                let mut max_depth = 25;
                let mut time_limit_ms = None;

                let mut i = 1;
                while i < tokens.len() {
                    match tokens[i] {
                        "depth" => {
                            if i + 1 < tokens.len() {
                                max_depth = tokens[i + 1].parse().unwrap_or(25);
                                i += 1;
                            }
                        }
                        "time" | "movetime" => {
                            if i + 1 < tokens.len() {
                                time_limit_ms = tokens[i + 1].parse().ok();
                                i += 1;
                            }
                        }
                        "wtime" | "btime" => {
                            if i + 1 < tokens.len() {
                                let time_left: u64 = tokens[i + 1].parse().unwrap_or(30000);
                                // Phân bổ khoảng 1/30 thời gian còn lại cho mỗi nước đi
                                time_limit_ms = Some((time_left / 30).max(50));
                                i += 1;
                            }
                        }
                        _ => {}
                    }
                    i += 1;
                }

                stop_signal.store(false, Ordering::Relaxed);

                let result = search_best_move(
                    &mut board,
                    &game_history,
                    max_depth,
                    time_limit_ms,
                    &model,
                    &mut tt,
                    Some(stop_signal.clone()),
                    true,
                );

                if let Some(m) = result.best_move {
                    let move_str = move_to_ucci_string(m);
                    println!("bestmove {}", move_str);
                } else {
                    println!("nobestmove");
                }
            }
            "stop" => {
                stop_signal.store(true, Ordering::Relaxed);
            }
            "quit" => {
                break;
            }
            "d" => {
                println!("Side to move: {:?}", board.side_to_move);
                for r in 0..10 {
                    for c in 0..9 {
                        let sq = Board::coord_to_square(r, c);
                        if let Some(p) = board.piece_at(sq) {
                            let mut ch = match p.piece_type {
                                PieceType::King => 'k',
                                PieceType::Advisor => 'a',
                                PieceType::Elephant => 'b',
                                PieceType::Horse => 'n',
                                PieceType::Rook => 'r',
                                PieceType::Cannon => 'c',
                                PieceType::Pawn => 'p',
                            };
                            if p.color == Color::Red {
                                ch = ch.to_ascii_uppercase();
                            }
                            print!("{} ", ch);
                        } else {
                            print!(". ");
                        }
                    }
                    println!();
                }
            }
            _ => {}
        }
    }
}
