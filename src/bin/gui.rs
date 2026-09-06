use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;

use macroquad::prelude::*;
use wisdom::ab_tt::TranspositionTableAB;
use wisdom::board::{Board, Color as PieceColor, HistoryEntry, PieceType, RepetitionResult};
use wisdom::nnue::NNUEModel;
use wisdom::r#move::Move;
use wisdom::search::{search_best_move_with_callback, SearchProgress};
use wisdom::ucci::{move_to_ucci_string, parse_ucci_move};

const SQUARE_SIZE: f32 = 60.0;
const OFFSET_X: f32 = 50.0;
const OFFSET_Y: f32 = 50.0;
const RADIUS: f32 = 25.0;

#[derive(PartialEq)]
enum GameMode {
    EngineVsPlayer,
    EngineVsEngine,
}

fn find_model_path() -> std::path::PathBuf {
    let candidates = [
        std::path::PathBuf::from("./wisdom_models/wisdom_nnue.bin"),
        std::path::PathBuf::from("wisdom_models/wisdom_nnue.bin"),
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

fn get_legal_moves(board: &mut Board) -> Vec<Move> {
    let moves = board.generate_all_moves();
    let moving_side = board.side_to_move;

    moves
        .into_iter()
        .filter(|&m| {
            let undo = board.make_move(m);
            let valid = !board.kings_facing() && !board.is_in_check(moving_side);
            board.unmake_move(m, undo);
            valid
        })
        .collect()
}

fn apply_move_to_game(board: &mut Board, m: Move, history: &mut Vec<HistoryEntry>) {
    let is_capture = !board.is_empty(m.to_sq());
    let piece = board.piece_at(m.from_sq()).unwrap();
    let is_reversible = !is_capture
        && (piece.piece_type != PieceType::Pawn || {
            let (from_row, _) = Board::square_to_coord(m.from_sq());
            let (to_row, _) = Board::square_to_coord(m.to_sq());
            from_row == to_row
        });
    let moving_side = board.side_to_move;

    let pre_threats = if is_reversible {
        board.get_unprotected_threats(moving_side)
    } else {
        0
    };

    board.make_move(m);

    let gives_check = board.is_in_check(board.side_to_move);

    let chased_set = if is_reversible && !gives_check {
        let post_threats = board.get_unprotected_threats(moving_side);
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
}

fn format_move(m: Move) -> String {
    move_to_ucci_string(m)
}

fn draw_board() {
    clear_background(Color::new(0.9, 0.8, 0.6, 1.0));

    for col in 0..9 {
        let x = OFFSET_X + col as f32 * SQUARE_SIZE;
        draw_line(x, OFFSET_Y, x, OFFSET_Y + 4.0 * SQUARE_SIZE, 2.0, BLACK);
        draw_line(
            x,
            OFFSET_Y + 5.0 * SQUARE_SIZE,
            x,
            OFFSET_Y + 9.0 * SQUARE_SIZE,
            2.0,
            BLACK,
        );
    }
    draw_line(
        OFFSET_X,
        OFFSET_Y + 4.0 * SQUARE_SIZE,
        OFFSET_X + 8.0 * SQUARE_SIZE,
        OFFSET_Y + 4.0 * SQUARE_SIZE,
        2.0,
        BLACK,
    );
    draw_line(
        OFFSET_X,
        OFFSET_Y + 5.0 * SQUARE_SIZE,
        OFFSET_X + 8.0 * SQUARE_SIZE,
        OFFSET_Y + 5.0 * SQUARE_SIZE,
        2.0,
        BLACK,
    );
    draw_line(
        OFFSET_X,
        OFFSET_Y + 4.0 * SQUARE_SIZE,
        OFFSET_X,
        OFFSET_Y + 5.0 * SQUARE_SIZE,
        2.0,
        BLACK,
    );
    draw_line(
        OFFSET_X + 8.0 * SQUARE_SIZE,
        OFFSET_Y + 4.0 * SQUARE_SIZE,
        OFFSET_X + 8.0 * SQUARE_SIZE,
        OFFSET_Y + 5.0 * SQUARE_SIZE,
        2.0,
        BLACK,
    );

    for row in 0..10 {
        let y = OFFSET_Y + row as f32 * SQUARE_SIZE;
        draw_line(OFFSET_X, y, OFFSET_X + 8.0 * SQUARE_SIZE, y, 2.0, BLACK);
    }

    // Palaces
    draw_line(
        OFFSET_X + 3.0 * SQUARE_SIZE,
        OFFSET_Y,
        OFFSET_X + 5.0 * SQUARE_SIZE,
        OFFSET_Y + 2.0 * SQUARE_SIZE,
        2.0,
        BLACK,
    );
    draw_line(
        OFFSET_X + 5.0 * SQUARE_SIZE,
        OFFSET_Y,
        OFFSET_X + 3.0 * SQUARE_SIZE,
        OFFSET_Y + 2.0 * SQUARE_SIZE,
        2.0,
        BLACK,
    );
    draw_line(
        OFFSET_X + 3.0 * SQUARE_SIZE,
        OFFSET_Y + 7.0 * SQUARE_SIZE,
        OFFSET_X + 5.0 * SQUARE_SIZE,
        OFFSET_Y + 9.0 * SQUARE_SIZE,
        2.0,
        BLACK,
    );
    draw_line(
        OFFSET_X + 5.0 * SQUARE_SIZE,
        OFFSET_Y + 7.0 * SQUARE_SIZE,
        OFFSET_X + 3.0 * SQUARE_SIZE,
        OFFSET_Y + 9.0 * SQUARE_SIZE,
        2.0,
        BLACK,
    );

    draw_text(
        "楚 河             漢 界",
        OFFSET_X + 1.5 * SQUARE_SIZE,
        OFFSET_Y + 4.6 * SQUARE_SIZE,
        30.0,
        BLACK,
    );
}

fn display_row(r: usize, human_color: PieceColor) -> usize {
    if human_color == PieceColor::Red {
        r
    } else {
        9 - r
    }
}
fn display_col(c: usize, human_color: PieceColor) -> usize {
    if human_color == PieceColor::Red {
        c
    } else {
        8 - c
    }
}

fn draw_pieces(
    board: &Board,
    selected_sq: Option<usize>,
    legal_moves: &[Move],
    human_color: PieceColor,
    font: Option<&Font>,
) {
    for row in 0..10 {
        for col in 0..9 {
            let sq = Board::coord_to_square(row, col);
            let d_row = display_row(row, human_color);
            let d_col = display_col(col, human_color);
            let x = OFFSET_X + d_col as f32 * SQUARE_SIZE;
            let y = OFFSET_Y + d_row as f32 * SQUARE_SIZE;

            if Some(sq) == selected_sq {
                draw_circle(x, y, RADIUS + 5.0, YELLOW);
            }

            if selected_sq.is_some() && legal_moves.iter().any(|m| m.to_sq() == sq) {
                draw_circle(x, y, 10.0, GREEN);
            }

            if let Some(piece) = board.piece_at(sq) {
                draw_circle(
                    x,
                    y,
                    RADIUS,
                    if piece.color == PieceColor::Red {
                        RED
                    } else {
                        BLACK
                    },
                );
                draw_circle_lines(x, y, RADIUS, 2.0, WHITE);

                let text = match piece.piece_type {
                    PieceType::King => {
                        if piece.color == PieceColor::Red {
                            "帥"
                        } else {
                            "將"
                        }
                    }
                    PieceType::Advisor => {
                        if piece.color == PieceColor::Red {
                            "仕"
                        } else {
                            "士"
                        }
                    }
                    PieceType::Elephant => {
                        if piece.color == PieceColor::Red {
                            "相"
                        } else {
                            "象"
                        }
                    }
                    PieceType::Horse => {
                        if piece.color == PieceColor::Red {
                            "傌"
                        } else {
                            "馬"
                        }
                    }
                    PieceType::Rook => {
                        if piece.color == PieceColor::Red {
                            "俥"
                        } else {
                            "車"
                        }
                    }
                    PieceType::Cannon => {
                        if piece.color == PieceColor::Red {
                            "炮"
                        } else {
                            "砲"
                        }
                    }
                    PieceType::Pawn => {
                        if piece.color == PieceColor::Red {
                            "兵"
                        } else {
                            "卒"
                        }
                    }
                };

                let text_size = measure_text(text, font, 30, 1.0);
                draw_text_ex(
                    text,
                    x - text_size.width / 2.0,
                    y + text_size.height / 2.0,
                    TextParams {
                        font,
                        font_size: 30,
                        color: WHITE,
                        ..Default::default()
                    },
                );
            }
        }
    }
}

fn draw_button(text: &str, x: f32, y: f32, w: f32, h: f32, active: bool, color: Color) -> bool {
    let bg_color = if active { GREEN } else { color };
    draw_rectangle(x, y, w, h, bg_color);
    draw_rectangle_lines(x, y, w, h, 2.0, BLACK);

    let text_size = measure_text(text, None, 20, 1.0);
    draw_text(
        text,
        x + (w - text_size.width) / 2.0,
        y + (h + text_size.height) / 2.0,
        20.0,
        WHITE,
    );

    if is_mouse_button_pressed(MouseButton::Left) {
        let (mx, my) = mouse_position();
        if mx >= x && mx <= x + w && my >= y && my <= y + h {
            return true;
        }
    }
    false
}

fn draw_eval_bar(eval: f32, x: f32, y: f32, w: f32, h: f32) {
    // eval in [-1, 1]: -1 = losing badly, +1 = winning
    // Bar background
    draw_rectangle(x, y, w, h, DARKGRAY);
    draw_rectangle_lines(x, y, w, h, 2.0, BLACK);

    // Fill from center based on eval
    let center_x = x + w / 2.0;
    let fill_width = eval.abs() * (w / 2.0);

    if eval >= 0.0 {
        // Green for positive (good for player)
        let green = Color::new(0.2, 0.8, 0.3, 1.0);
        draw_rectangle(center_x, y + 2.0, fill_width, h - 4.0, green);
    } else {
        // Red for negative (bad for player)
        let red = Color::new(0.9, 0.2, 0.2, 1.0);
        draw_rectangle(center_x - fill_width, y + 2.0, fill_width, h - 4.0, red);
    }

    // Center line
    draw_line(center_x, y, center_x, y + h, 2.0, WHITE);

    // Eval text
    let eval_text = format!("{:+.3}", eval);
    let text_color = if eval >= 0.0 {
        Color::new(0.1, 0.6, 0.1, 1.0)
    } else {
        Color::new(0.8, 0.1, 0.1, 1.0)
    };
    draw_text(
        &eval_text,
        x + w + 10.0,
        y + h / 2.0 + 5.0,
        20.0,
        text_color,
    );
}

/// Replays moves from startpos to rebuild board and history
fn replay_moves_from_startpos(
    board: &mut Board,
    game_history: &mut Vec<HistoryEntry>,
    moves_uci: &[String],
) {
    board.set_initial_position();
    game_history.clear();
    for move_str in moves_uci {
        if let Some(m) = parse_ucci_move(board, move_str) {
            apply_move_to_game(board, m, game_history);
        }
    }
}

fn window_conf() -> Conf {
    Conf {
        window_title: "Wisdom Engine - Xiangqi (NNUE Alpha-Beta)".to_owned(),
        window_width: 900,
        window_height: 700,
        ..Default::default()
    }
}

enum EngineCommand {
    StartSearch {
        board: Board,
        history: Vec<HistoryEntry>,
        depth: usize,
    },
    Stop,
}

enum EngineMessage {
    Progress(SearchProgress),
    BestMove(Move),
    NoMove,
}

#[macroquad::main(window_conf)]
async fn main() {
    // Attempt to load custom font
    let mut custom_font: Option<Font> = None;
    match load_ttf_font("assets/font.ttf").await {
        Ok(f) => {
            println!("GUI: Successfully loaded assets/font.ttf");
            custom_font = Some(f);
        }
        Err(_) => {
            println!(
                "GUI: Could not load assets/font.ttf, falling back to default. Characters might not render correctly."
            );
        }
    }

    // --- Game State ---
    let mut board = Board::new();
    board.set_initial_position();

    let mut game_history: Vec<HistoryEntry> = Vec::new();
    let mut game_moves_uci: Vec<String> = Vec::new();

    let mut selected_sq: Option<usize> = None;
    let mut legal_moves = Vec::new();
    let mut game_over = false;
    let mut game_over_message = String::from("GAME OVER");

    // UI State
    let mut game_mode = GameMode::EngineVsPlayer;
    let mut human_color = PieceColor::Red;
    let mut search_depth: usize = 12;
    let mut current_eval: Option<String> = Some("Ready".to_string());
    let mut current_stats: Option<String> = None;
    let mut engine_pv: Vec<String> = Vec::new();
    let mut current_eval_score: Option<f32> = None; // Advantage in [-1, 1] from player's perspective

    // Find and load Wisdom NNUE model
    let model_path = find_model_path();
    println!("GUI: Loading Wisdom NNUE model from {:?}...", model_path);
    let model = match NNUEModel::load_from_file(&model_path) {
        Ok(m) => Arc::new(m),
        Err(e) => {
            eprintln!("GUI Error: Failed to load NNUE model from {:?}: {}", model_path, e);
            panic!("Cannot start GUI without NNUE model: {}", e);
        }
    };
    println!("GUI: Successfully loaded Wisdom NNUE model! ⚡️🚀");

    // Spawn Background Worker Thread for Search
    let (cmd_tx, cmd_rx) = mpsc::channel::<EngineCommand>();
    let (resp_tx, resp_rx) = mpsc::channel::<EngineMessage>();
    let stop_signal = Arc::new(AtomicBool::new(false));

    let thread_model = model.clone();
    let thread_stop = stop_signal.clone();

    thread::spawn(move || {
        let mut tt = TranspositionTableAB::new(64); // 64 MB TT table

        while let Ok(cmd) = cmd_rx.recv() {
            match cmd {
                EngineCommand::StartSearch { mut board, history, depth } => {
                    thread_stop.store(false, Ordering::Relaxed);
                    let tx_clone = resp_tx.clone();
                    let sig = thread_stop.clone();

                    let res = search_best_move_with_callback(
                        &mut board,
                        &history,
                        depth,
                        None,
                        &thread_model,
                        &mut tt,
                        Some(sig),
                        |prog| {
                            tx_clone.send(EngineMessage::Progress(prog.clone())).unwrap_or(());
                        },
                    );

                    if let Some(m) = res.best_move {
                        resp_tx.send(EngineMessage::BestMove(m)).unwrap_or(());
                    } else {
                        resp_tx.send(EngineMessage::NoMove).unwrap_or(());
                    }
                }
                EngineCommand::Stop => {
                    thread_stop.store(true, Ordering::Relaxed);
                }
            }
        }
    });

    let mut is_engine_thinking = false;

    loop {
        draw_board();
        draw_pieces(
            &board,
            selected_sq,
            &legal_moves,
            human_color,
            custom_font.as_ref(),
        );

        // --- DRAW CONTROL PANEL ---
        let panel_x = 620.0;
        let mut py = OFFSET_Y;

        draw_text("Wisdom NNUE GUI", panel_x, py, 28.0, BLACK);
        py += 40.0;

        // Reset Button
        if draw_button("Reset Game", panel_x, py, 95.0, 40.0, false, BLUE) {
            if is_engine_thinking {
                stop_signal.store(true, Ordering::Relaxed);
                cmd_tx.send(EngineCommand::Stop).unwrap_or(());
                is_engine_thinking = false;
            }
            board.set_initial_position();
            game_history.clear();
            game_moves_uci.clear();
            selected_sq = None;
            legal_moves.clear();
            game_over = false;
            game_over_message = "GAME OVER".into();
            current_eval = Some("Ready".to_string());
            current_stats = None;
            current_eval_score = None;
            engine_pv.clear();
        }

        // Undo Button (only in Player vs Engine mode)
        let can_undo = !is_engine_thinking
            && game_mode == GameMode::EngineVsPlayer
            && game_moves_uci.len() >= 2
            && !game_over;
        let undo_color = if can_undo { ORANGE } else { LIGHTGRAY };
        if can_undo && draw_button("Undo", panel_x + 105.0, py, 95.0, 40.0, false, undo_color) {
            game_moves_uci.pop();
            game_moves_uci.pop();
            replay_moves_from_startpos(&mut board, &mut game_history, &game_moves_uci);
            selected_sq = None;
            legal_moves.clear();
            current_eval = Some("Move undone".to_string());
            current_stats = None;
            current_eval_score = None;
            engine_pv.clear();
        } else if !can_undo {
            draw_button("Undo", panel_x + 105.0, py, 95.0, 40.0, false, LIGHTGRAY);
        }
        py += 55.0;

        // Game Mode Toggle
        draw_text("Game Mode:", panel_x, py, 20.0, BLACK);
        py += 25.0;
        if !is_engine_thinking
            && draw_button(
                "Player vs Engine",
                panel_x,
                py,
                180.0,
                30.0,
                game_mode == GameMode::EngineVsPlayer,
                GRAY,
            )
        {
            game_mode = GameMode::EngineVsPlayer;
            board.set_initial_position();
            game_history.clear();
            game_moves_uci.clear();
            selected_sq = None;
            legal_moves.clear();
            game_over = false;
            game_over_message = "GAME OVER".into();
            current_eval = Some("Ready".to_string());
            current_stats = None;
            current_eval_score = None;
            engine_pv.clear();
        }
        py += 38.0;
        if !is_engine_thinking
            && draw_button(
                "Engine vs Engine",
                panel_x,
                py,
                180.0,
                30.0,
                game_mode == GameMode::EngineVsEngine,
                GRAY,
            )
        {
            game_mode = GameMode::EngineVsEngine;
            board.set_initial_position();
            game_history.clear();
            game_moves_uci.clear();
            selected_sq = None;
            legal_moves.clear();
            game_over = false;
            game_over_message = "GAME OVER".into();
            current_eval = Some("Ready".to_string());
            current_stats = None;
            current_eval_score = None;
            engine_pv.clear();
        }
        py += 38.0;

        // Side Selection
        if game_mode == GameMode::EngineVsPlayer {
            draw_text("Player Side:", panel_x, py, 20.0, BLACK);
            py += 25.0;
            if !is_engine_thinking
                && draw_button(
                    "Play Red",
                    panel_x,
                    py,
                    80.0,
                    30.0,
                    human_color == PieceColor::Red,
                    GRAY,
                )
            {
                human_color = PieceColor::Red;
                board.set_initial_position();
                game_history.clear();
                game_moves_uci.clear();
                selected_sq = None;
                legal_moves.clear();
                game_over = false;
                game_over_message = "GAME OVER".into();
                current_eval = Some("Ready".to_string());
                current_stats = None;
                current_eval_score = None;
                engine_pv.clear();
            }
            if !is_engine_thinking
                && draw_button(
                    "Play Black",
                    panel_x + 90.0,
                    py,
                    90.0,
                    30.0,
                    human_color == PieceColor::Black,
                    GRAY,
                )
            {
                human_color = PieceColor::Black;
                board.set_initial_position();
                game_history.clear();
                game_moves_uci.clear();
                selected_sq = None;
                legal_moves.clear();
                game_over = false;
                game_over_message = "GAME OVER".into();
                current_eval = Some("Ready".to_string());
                current_stats = None;
                current_eval_score = None;
                engine_pv.clear();
            }
            py += 38.0;
        }

        // Search Depth Control
        draw_text(
            &format!("Search Depth: {} plies", search_depth),
            panel_x,
            py,
            18.0,
            BLACK,
        );
        py += 22.0;
        if !is_engine_thinking && draw_button("-", panel_x, py, 40.0, 28.0, false, GRAY) {
            if search_depth > 4 {
                search_depth -= 1;
            }
        }
        if !is_engine_thinking && draw_button("+", panel_x + 50.0, py, 40.0, 28.0, false, GRAY) {
            if search_depth < 25 {
                search_depth += 1;
            }
        }
        py += 40.0;

        // Advantage Bar Display
        if let Some(eval_score) = current_eval_score {
            draw_text("Advantage:", panel_x, py, 18.0, BLACK);
            py += 22.0;
            draw_eval_bar(eval_score, panel_x, py, 180.0, 18.0);
            py += 28.0;
        }

        // Eval Text Display
        if let Some(ref eval_str) = current_eval {
            draw_text(eval_str, panel_x, py, 18.0, DARKGREEN);
            py += 22.0;
        }

        // Stats Display (Nodes, NPS)
        if let Some(ref stats_str) = current_stats {
            draw_text(stats_str, panel_x, py, 15.0, Color::new(0.1, 0.3, 0.7, 1.0));
            py += 22.0;
        }

        // Predicted Line (PV)
        if !engine_pv.is_empty() {
            draw_text("Predicted Line (PV):", panel_x, py, 17.0, BLACK);
            py += 20.0;
            for line in &engine_pv {
                draw_text(line, panel_x, py, 15.0, DARKGRAY);
                py += 17.0;
            }
        }

        if game_over {
            draw_text(&game_over_message, OFFSET_X, OFFSET_Y / 2.0, 30.0, RED);
        }

        // Non-blocking wait for Engine reply
        while let Ok(msg) = resp_rx.try_recv() {
            match msg {
                EngineMessage::Progress(prog) => {
                    let player_score = if human_color == PieceColor::Red {
                        prog.score
                    } else {
                        -prog.score
                    };
                    let norm = (player_score as f32 / 400.0).clamp(-1.0, 1.0);
                    current_eval_score = Some(norm);

                    current_eval = Some(format!("Depth: {} | Score: {:+4} cp", prog.depth, prog.score));
                    current_stats = Some(format!("{} nodes | {} NPS", prog.nodes, prog.nps));

                    engine_pv.clear();
                    let moves: Vec<&str> = prog.pv.split_whitespace().collect();
                    for (i, m) in moves.iter().take(5).enumerate() {
                        engine_pv.push(format!("{}. {}", i + 1, m));
                    }
                }
                EngineMessage::BestMove(m) => {
                    apply_move_to_game(&mut board, m, &mut game_history);
                    game_moves_uci.push(format_move(m));

                    // GUI Repetition check after Engine plays
                    if game_history.len() >= 4 {
                        match board.judge_repetition(&game_history, game_history.len(), 2) {
                            RepetitionResult::Win => {
                                game_over = true;
                                game_over_message = "Engine Violation: You Win!".into();
                                current_eval = Some("WON".to_string());
                            }
                            RepetitionResult::Loss => {
                                game_over = true;
                                game_over_message = "Rule Violation: Engine Wins!".into();
                                current_eval = Some("LOST".to_string());
                            }
                            RepetitionResult::Draw => {
                                game_over = true;
                                game_over_message = "Draw by Repetition!".into();
                                current_eval = Some("DRAW".to_string());
                            }
                            RepetitionResult::Undecided => {
                                if get_legal_moves(&mut board).is_empty() {
                                    game_over = true;
                                    game_over_message = "Checkmate!".into();
                                    current_eval = Some("CHECKMATE".to_string());
                                }
                            }
                        }
                    } else {
                        if get_legal_moves(&mut board).is_empty() {
                            game_over = true;
                            game_over_message = "Checkmate!".into();
                            current_eval = Some("CHECKMATE".to_string());
                        }
                    }
                    is_engine_thinking = false;
                }
                EngineMessage::NoMove => {
                    game_over = true;
                    game_over_message = "Checkmate / Stalemate!".into();
                    current_eval = Some("CHECKMATE".to_string());
                    is_engine_thinking = false;
                }
            }
        }

        // --- GAME LOGIC ---
        if !game_over && !is_engine_thinking {
            let is_human_turn =
                game_mode == GameMode::EngineVsPlayer && board.side_to_move == human_color;

            if is_human_turn {
                if is_mouse_button_pressed(MouseButton::Left) {
                    let (mx, my) = mouse_position();
                    let d_col =
                        ((mx - OFFSET_X + SQUARE_SIZE / 2.0) / SQUARE_SIZE).floor() as isize;
                    let d_row =
                        ((my - OFFSET_Y + SQUARE_SIZE / 2.0) / SQUARE_SIZE).floor() as isize;

                    if d_col >= 0 && d_col < 9 && d_row >= 0 && d_row < 10 {
                        let c = if human_color == PieceColor::Red {
                            d_col as usize
                        } else {
                            8 - (d_col as usize)
                        };
                        let r = if human_color == PieceColor::Red {
                            d_row as usize
                        } else {
                            9 - (d_row as usize)
                        };
                        let sq = Board::coord_to_square(r, c);

                        if selected_sq.is_some() {
                            if let Some(&m) = legal_moves.iter().find(|m| m.to_sq() == sq) {
                                apply_move_to_game(&mut board, m, &mut game_history);
                                game_moves_uci.push(format_move(m));
                                selected_sq = None;
                                legal_moves.clear();

                                if game_history.len() >= 4 {
                                    match board.judge_repetition(
                                        &game_history,
                                        game_history.len(),
                                        2,
                                    ) {
                                        RepetitionResult::Win => {
                                            game_over = true;
                                            game_over_message = "Rule Violation: You Lose!".into();
                                            current_eval = Some("LOST".to_string());
                                        }
                                        RepetitionResult::Loss => {
                                            game_over = true;
                                            game_over_message =
                                                "Opponent Violation: You Win!".into();
                                            current_eval = Some("WON".to_string());
                                        }
                                        RepetitionResult::Draw => {
                                            game_over = true;
                                            game_over_message = "Draw by Repetition!".into();
                                            current_eval = Some("DRAW".to_string());
                                        }
                                        RepetitionResult::Undecided => {
                                            let all = get_legal_moves(&mut board);
                                            if all.is_empty() {
                                                game_over = true;
                                                game_over_message = "Checkmate!".into();
                                                current_eval = Some("CHECKMATE".to_string());
                                            } else {
                                                current_eval =
                                                    Some("Wisdom NNUE Thinking...".to_string());
                                            }
                                        }
                                    }
                                } else {
                                    let all = get_legal_moves(&mut board);
                                    if all.is_empty() {
                                        game_over = true;
                                        game_over_message = "Checkmate!".into();
                                        current_eval = Some("CHECKMATE".to_string());
                                    } else {
                                        current_eval = Some("Wisdom NNUE Thinking...".to_string());
                                    }
                                }
                            } else {
                                if let Some(piece) = board.piece_at(sq) {
                                    if piece.color == human_color {
                                        selected_sq = Some(sq);
                                        legal_moves = get_legal_moves(&mut board)
                                            .into_iter()
                                            .filter(|m| m.from_sq() == sq)
                                            .collect();
                                    } else {
                                        selected_sq = None;
                                        legal_moves.clear();
                                    }
                                } else {
                                    selected_sq = None;
                                    legal_moves.clear();
                                }
                            }
                        } else {
                            if let Some(piece) = board.piece_at(sq) {
                                if piece.color == human_color {
                                    selected_sq = Some(sq);
                                    legal_moves = get_legal_moves(&mut board)
                                        .into_iter()
                                        .filter(|m| m.from_sq() == sq)
                                        .collect();
                                }
                            }
                        }
                    }
                }
            } else {
                // ENGINE TURN: Trigger Search command
                let all_moves = get_legal_moves(&mut board);
                if all_moves.is_empty() {
                    game_over = true;
                    game_over_message = "Checkmate!".into();
                    current_eval = Some("CHECKMATE".to_string());
                } else {
                    current_eval = Some("Wisdom NNUE Thinking...".to_string());
                    is_engine_thinking = true;
                    cmd_tx.send(EngineCommand::StartSearch {
                        board: board.clone(),
                        history: game_history.clone(),
                        depth: search_depth,
                    }).unwrap_or(());
                }
            }
        }

        next_frame().await
    }
}
