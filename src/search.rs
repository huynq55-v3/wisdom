use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::ab_tt::{
    TranspositionTableAB, TT_FLAG_EXACT, TT_FLAG_LOWER, TT_FLAG_UPPER, VALUE_MATE,
    VALUE_MATE_THRESHOLD,
};
use crate::board::{Board, Color, HistoryEntry, PieceType};
use crate::nnue::{Accumulator, NNUEModel};
use crate::r#move::Move;

pub const MAX_PLY: usize = 64;
pub const VALUE_INFINITE: i16 = 30000;
pub const VALUE_DRAW: i16 = 0;

// Giá trị tương đối của các quân cờ cho thuật toán sắp xếp nước đi MVV-LVA
const PIECE_VALS: [i32; 7] = [
    100,  // Pawn
    200,  // Advisor
    200,  // Elephant
    400,  // Horse
    450,  // Cannon
    900,  // Rook
    10000 // King
];

#[inline(always)]
fn get_piece_value(pt: PieceType) -> i32 {
    match pt {
        PieceType::Pawn => PIECE_VALS[0],
        PieceType::Advisor => PIECE_VALS[1],
        PieceType::Elephant => PIECE_VALS[2],
        PieceType::Horse => PIECE_VALS[3],
        PieceType::Cannon => PIECE_VALS[4],
        PieceType::Rook => PIECE_VALS[5],
        PieceType::King => PIECE_VALS[6],
    }
}

pub struct SearchState<'a> {
    pub nnue: &'a NNUEModel,
    pub tt: &'a mut TranspositionTableAB,
    pub acc_stack: [Accumulator; MAX_PLY + 1],
    pub eval_stack: [i16; MAX_PLY + 1],
    pub killers: [[Option<Move>; 2]; MAX_PLY + 1],
    pub history: [[[i32; 90]; 90]; 2], // [color][from_dense][to_dense]
    pub nodes: u64,
    pub start_time: Instant,
    pub time_limit_ms: Option<u64>,
    pub stop_signal: Option<Arc<AtomicBool>>,
    pub stopped: bool,
    pub game_history: Vec<u64>,
}

impl<'a> SearchState<'a> {
    pub fn new(
        nnue: &'a NNUEModel,
        tt: &'a mut TranspositionTableAB,
        stop_signal: Option<Arc<AtomicBool>>,
    ) -> Self {
        Self {
            nnue,
            tt,
            acc_stack: [Accumulator::new(&nnue.ft_bias); MAX_PLY + 1],
            eval_stack: [0; MAX_PLY + 1],
            killers: [[None; 2]; MAX_PLY + 1],
            history: [[[0; 90]; 90]; 2],
            nodes: 0,
            start_time: Instant::now(),
            time_limit_ms: None,
            stop_signal,
            stopped: false,
            game_history: Vec::new(),
        }
    }

    #[inline(always)]
    pub fn check_time(&mut self) -> bool {
        if self.stopped {
            return true;
        }

        if (self.nodes & 2047) == 0 {
            if let Some(ref sig) = self.stop_signal {
                if sig.load(Ordering::Relaxed) {
                    self.stopped = true;
                    return true;
                }
            }

            if let Some(limit) = self.time_limit_ms {
                if self.start_time.elapsed().as_millis() as u64 >= limit {
                    self.stopped = true;
                    return true;
                }
            }
        }

        false
    }
}

pub struct SearchResult {
    pub best_move: Option<Move>,
    pub score: i16,
    pub depth: usize,
    pub nodes: u64,
    pub time_ms: u64,
    pub nps: u64,
}

// =====================================================================
// 1. SẮP XẾP NƯỚC ĐI (MOVE ORDERING - MVV-LVA, KILLERS, HISTORY)
// =====================================================================
fn score_move(
    board: &Board,
    m: Move,
    tt_move: Option<Move>,
    ply: usize,
    state: &SearchState,
) -> i32 {
    // 1. Nước đi từ Transposition Table ưu tiên số 1
    if let Some(tm) = tt_move {
        if tm.0 == m.0 {
            return 2_000_000;
        }
    }

    let to_sq = m.to_sq();
    let from_sq = m.from_sq();
    let moving_piece = board.piece_at(from_sq).unwrap();

    // 2. Nước bắt quân (MVV-LVA: Nạn nhân giá trị cao nhất - Kẻ tấn công giá trị thấp nhất)
    if let Some(captured) = board.piece_at(to_sq) {
        let victim_val = get_piece_value(captured.piece_type);
        let attacker_val = get_piece_value(moving_piece.piece_type);
        return 1_000_000 + victim_val * 10 - attacker_val;
    }

    // 3. Killer Moves (2 nước tĩnh từng gây beta cutoff ở cùng độ sâu ply)
    if let Some(k1) = state.killers[ply][0] {
        if k1.0 == m.0 {
            return 900_000;
        }
    }
    if let Some(k2) = state.killers[ply][1] {
        if k2.0 == m.0 {
            return 800_000;
        }
    }

    // 4. History Heuristic (Điểm lịch sử thành công của nước cờ)
    let color_idx = if board.side_to_move == Color::Red { 0 } else { 1 };
    let from_dense = Board::square_to_dense(from_sq);
    let to_dense = Board::square_to_dense(to_sq);

    state.history[color_idx][from_dense][to_dense]
}

// =====================================================================
// BẢNG GIẢM ĐỘ SÂU LOGARITHM (LMR - LATE MOVE REDUCTION)
// =====================================================================
#[inline(always)]
fn get_reduction(depth: i8, move_count: usize, is_quiet: bool, in_check: bool, improving: bool, move_score: i32) -> i8 {
    if depth < 3 || move_count < 2 || !is_quiet || in_check {
        return 0;
    }

    let d = depth as f32;
    let m = move_count as f32;

    // Công thức logarit chuẩn Stockfish: R = 0.75 + ln(depth) * ln(moveCount) / 2.25
    let mut r = (0.75 + d.ln() * m.ln() / 2.25) as i8;

    // Không cải thiện thế cờ (not improving): giảm sâu hơn 1 ply
    if !improving {
        r += 1;
    }

    // Giảm nhẹ reduction nếu nước đi có điểm sắp xếp cao (Killer move hoặc History tốt)
    if move_score >= 800_000 {
        r = (r - 1).max(0);
    }

    r.min(depth - 1).max(0)
}

// =====================================================================
// 2. TÌM KIẾM TĨNH (QUIESCENCE SEARCH - CHỐNG HORIZON EFFECT)
// =====================================================================
fn quiescence_search(
    board: &mut Board,
    ply: usize,
    mut alpha: i16,
    beta: i16,
    state: &mut SearchState,
) -> i16 {
    state.nodes += 1;
    if state.check_time() || ply >= MAX_PLY {
        return state.nnue.evaluate_cp(board, &state.acc_stack[ply]);
    }

    // 1. Stand-pat: Thẩm định điểm tĩnh hiện tại bằng NNUE
    let stand_pat = state.nnue.evaluate_cp(board, &state.acc_stack[ply]);

    if stand_pat >= beta {
        return beta; // Beta cutoff ngay lập tức
    }
    if stand_pat > alpha {
        alpha = stand_pat;
    }

    // 2. Chỉ sinh các nước cờ bắt quân
    let mut captures = board.generate_captures();
    let moving_side = board.side_to_move;

    // Sắp xếp các nước bắt quân theo MVV-LVA
    captures.sort_unstable_by_key(|m| {
        let victim_val = board.piece_at(m.to_sq()).map(|p| get_piece_value(p.piece_type)).unwrap_or(0);
        let attacker_val = board.piece_at(m.from_sq()).map(|p| get_piece_value(p.piece_type)).unwrap_or(0);
        -(victim_val * 10 - attacker_val)
    });

    for m in captures {
        let to_sq = m.to_sq();
        let from_sq = m.from_sq();
        let captured = board.piece_at(to_sq);
        let moving_piece = board.piece_at(from_sq).unwrap();

        // Delta Pruning: nếu nước ăn quân lớn nhất vẫn không giúp vượt qua alpha, bỏ qua
        if let Some(cap) = captured {
            let cap_val = get_piece_value(cap.piece_type) as i16;
            if stand_pat.saturating_add(cap_val).saturating_add(200) < alpha {
                continue;
            }
        }

        // Kiểm tra tính hợp lệ trong vòng lặp (Lazy legality checking)
        let undo = board.make_move(m);
        if board.kings_facing() || board.is_in_check(moving_side) {
            board.unmake_move(m, undo);
            continue;
        }

        // Cập nhật vi phân Accumulator
        state.acc_stack[ply + 1] = state.acc_stack[ply];
        state.acc_stack[ply + 1].update_move(board, moving_piece, from_sq, to_sq, captured, state.nnue);

        let score = -quiescence_search(board, ply + 1, -beta, -alpha, state);
        board.unmake_move(m, undo);

        if state.stopped {
            return 0;
        }

        if score >= beta {
            return beta;
        }
        if score > alpha {
            alpha = score;
        }
    }

    alpha
}

// =====================================================================
// 3. THUẬT TOÁN PVS / NEGAMAX CHÍNH (VỚI TOÀN BỘ KỸ THUẬT STOCKFISH)
// =====================================================================
pub fn negamax_pvs(
    board: &mut Board,
    ply: usize,
    mut depth: i8,
    mut alpha: i16,
    beta: i16,
    can_null: bool,
    state: &mut SearchState,
) -> i16 {
    state.nodes += 1;
    if state.check_time() {
        return 0;
    }

    // Lá cây tìm kiếm: chuyển sang Quiescence Search
    if depth <= 0 || ply >= MAX_PLY {
        return quiescence_search(board, ply, alpha, beta, state);
    }

    let is_root = ply == 0;
    let in_check = board.is_in_check(board.side_to_move);

    // Mở rộng tìm kiếm nếu đang bị chiếu (Check Extension chỉ khi depth > 0)
    if in_check && depth < 20 {
        depth += 1;
    }

    // 1. Kiểm tra lặp cờ (Repetition detection)
    if !is_root {
        let mut rep_count = 0;
        for &past_hash in state.game_history.iter().rev() {
            if past_hash == board.zobrist_key {
                rep_count += 1;
                if rep_count >= 1 {
                    return VALUE_DRAW; // Hòa do lặp nước
                }
            }
        }
    }

    // 2. Tra cứu Transposition Table (TT Cutoff)
    let (tt_move, tt_cutoff) = state.tt.probe(board.zobrist_key, ply, depth, alpha, beta);
    if !is_root {
        if let Some(score) = tt_cutoff {
            return score;
        }
    }

    let static_eval = state.nnue.evaluate_cp(board, &state.acc_stack[ply]);
    state.eval_stack[ply] = static_eval;

    let improving = if ply >= 2 {
        static_eval > state.eval_stack[ply - 2]
    } else {
        true
    };

    // 3. Reverse Futility Pruning (Static NMP): Ở độ sâu thấp nếu điểm tĩnh áp đảo beta
    if !is_root && !in_check && depth <= 7 {
        let rfp_margin = if improving { 70 * (depth as i16) } else { 90 * (depth as i16) };
        if static_eval - rfp_margin >= beta {
            return static_eval;
        }
    }

    // 4. Null Move Pruning (NMP): Nhường nước đi nếu thế cờ đang vượt trội
    if !is_root && can_null && !in_check && depth >= 3 && static_eval >= beta {
        let r = 3 + depth / 4 + (((static_eval - beta) / 200).clamp(0, 3) as i8);
        let null_depth = (depth - 1 - r).max(0);

        board.side_to_move = board.side_to_move.opposite();
        board.zobrist_key ^= crate::zobrist::ZOBRIST.side;
        state.acc_stack[ply + 1] = state.acc_stack[ply];

        let null_score = -negamax_pvs(
            board,
            ply + 1,
            null_depth,
            -beta,
            -beta + 1,
            false,
            state,
        );

        board.side_to_move = board.side_to_move.opposite();
        board.zobrist_key ^= crate::zobrist::ZOBRIST.side;

        if null_score >= beta {
            return if null_score >= VALUE_MATE_THRESHOLD { beta } else { null_score };
        }
    }

    // Internal Iterative Reductions (IIR)
    if !is_root && depth >= 6 && tt_move.is_none() {
        depth -= 1;
    }

    // 5. Sinh và Sắp xếp tất cả nước đi hợp lệ
    let moves = board.generate_all_moves();
    let moving_side = board.side_to_move;

    let mut scored_moves: Vec<(Move, i32)> = moves
        .into_iter()
        .map(|m| {
            let score = score_move(board, m, tt_move, ply, state);
            (m, score)
        })
        .collect();

    scored_moves.sort_unstable_by(|a, b| b.1.cmp(&a.1));

    let mut best_move = None;
    let mut best_score = -VALUE_INFINITE;
    let mut tt_flag = TT_FLAG_UPPER;
    let mut moves_searched = 0;
    let mut legal_moves_count = 0;

    state.game_history.push(board.zobrist_key);

    for (m, move_score) in &scored_moves {
        let m = *m;
        let from_sq = m.from_sq();
        let to_sq = m.to_sq();
        let captured = board.piece_at(to_sq);
        let moving_piece = board.piece_at(from_sq).unwrap();
        let is_quiet = captured.is_none();

        // 1. Late Move Pruning (LMP): Cắt tỉa các nước tĩnh sau một ngưỡng số nước ở độ sâu thấp
        if !is_root && depth <= 8 && !in_check && is_quiet {
            let lmp_limit = (3 + (depth as usize) * (depth as usize)) / (if improving { 1 } else { 2 });
            if moves_searched >= lmp_limit {
                continue;
            }
        }

        // 2. Futility Pruning (FP): Cắt tỉa nước tĩnh nếu điểm tĩnh + biên độ không thể chạm tới alpha
        if !is_root && depth <= 7 && !in_check && is_quiet {
            let futility_margin = (if improving { 100 } else { 80 }) * (depth as i16);
            if static_eval + futility_margin <= alpha {
                continue;
            }
        }

        // Thực hiện nước đi và kiểm tra tính hợp lệ
        let undo = board.make_move(m);
        if board.kings_facing() || board.is_in_check(moving_side) {
            board.unmake_move(m, undo);
            continue;
        }

        legal_moves_count += 1;

        // Cập nhật vi phân Accumulator
        state.acc_stack[ply + 1] = state.acc_stack[ply];
        state.acc_stack[ply + 1].update_move(board, moving_piece, from_sq, to_sq, captured, state.nnue);

        let mut score;

        if moves_searched == 0 {
            // Nước đầu tiên (PV move): Tìm kiếm với cửa sổ đầy đủ [alpha, beta]
            score = -negamax_pvs(board, ply + 1, depth - 1, -beta, -alpha, true, state);
        } else {
            // Các nước tiếp theo: Áp dụng LMR (Late Move Reductions) cho các nước tĩnh phía sau
            let reduction = get_reduction(depth, moves_searched, is_quiet, in_check, improving, *move_score);
            let search_depth = (depth - 1 - reduction).max(0);

            // Tìm kiếm với cửa sổ tối thiểu (Zero-Window: [-alpha-1, -alpha])
            score = -negamax_pvs(
                board,
                ply + 1,
                search_depth,
                -alpha - 1,
                -alpha,
                true,
                state,
            );

            // Nếu thất bại (score > alpha), tìm kiếm lại với độ sâu đầy đủ
            if score > alpha && (reduction > 0 || score < beta) {
                score = -negamax_pvs(board, ply + 1, depth - 1, -beta, -alpha, true, state);
            }
        }

        board.unmake_move(m, undo);
        moves_searched += 1;

        if state.stopped {
            state.game_history.pop();
            return 0;
        }

        if score > best_score {
            best_score = score;
            best_move = Some(m);
        }

        if score > alpha {
            alpha = score;
            tt_flag = TT_FLAG_EXACT;
        }

        // Beta Cutoff (Cắt tỉa nhánh)
        if alpha >= beta {
            tt_flag = TT_FLAG_LOWER;

            // Cập nhật Killer Moves & History Heuristic cho nước đi tĩnh
            if is_quiet {
                state.killers[ply][1] = state.killers[ply][0];
                state.killers[ply][0] = Some(m);

                let color_idx = if moving_side == Color::Red { 0 } else { 1 };
                let from_dense = Board::square_to_dense(from_sq);
                let to_dense = Board::square_to_dense(to_sq);
                let bonus = (depth as i32) * (depth as i32);
                state.history[color_idx][from_dense][to_dense] += bonus;

                // History Malus cho các nước tĩnh trước đó
                for (prev_m, _) in &scored_moves[..moves_searched.min(scored_moves.len())] {
                    if board.piece_at(prev_m.to_sq()).is_none() && prev_m.0 != m.0 {
                        let p_from = Board::square_to_dense(prev_m.from_sq());
                        let p_to = Board::square_to_dense(prev_m.to_sq());
                        state.history[color_idx][p_from][p_to] -= bonus / 2;
                    }
                }
            }

            break;
        }
    }

    state.game_history.pop();

    // Nếu không có nước đi hợp lệ nào (Bị chiếu bí hoặc Khốn cờ)
    if legal_moves_count == 0 {
        return -VALUE_MATE + (ply as i16);
    }

    // Lưu vào Transposition Table
    state.tt.record(
        board.zobrist_key,
        ply,
        depth,
        best_score,
        tt_flag,
        best_move,
    );

    best_score
}

// =====================================================================
// TRÍCH XUẤT NƯỚC ĐI TỐI ƯU (PV - PRINCIPAL VARIATION) TỪ TT
// =====================================================================
pub fn extract_pv_string(board: &mut Board, tt: &TranspositionTableAB, max_len: usize) -> String {
    let mut pv_moves = Vec::new();
    let mut undos = Vec::new();
    let mut visited = std::collections::HashSet::new();

    for _ in 0..max_len {
        let (tt_move, _) = tt.probe(board.zobrist_key, 0, 0, -VALUE_INFINITE, VALUE_INFINITE);
        if let Some(m) = tt_move {
            if visited.contains(&board.zobrist_key) {
                break;
            }
            visited.insert(board.zobrist_key);

            let undo = board.make_move(m);
            if board.kings_facing() || board.is_in_check(board.side_to_move.opposite()) {
                board.unmake_move(m, undo);
                break;
            }
            pv_moves.push(m);
            undos.push((m, undo));
        } else {
            break;
        }
    }

    for (m, undo) in undos.into_iter().rev() {
        board.unmake_move(m, undo);
    }

    pv_moves
        .iter()
        .map(|m| crate::ucci::move_to_ucci_string(*m))
        .collect::<Vec<_>>()
        .join(" ")
}

#[derive(Clone, Debug)]
pub struct SearchProgress {
    pub depth: usize,
    pub score: i16,
    pub time_ms: u64,
    pub nodes: u64,
    pub nps: u64,
    pub pv: String,
    pub best_move: Option<Move>,
}

// =====================================================================
// 4. DUYỆT SÂU DẦN (ITERATIVE DEEPENING & ASPIRATION WINDOWS)
// =====================================================================
pub fn search_best_move_with_callback<F>(
    board: &mut Board,
    history: &[HistoryEntry],
    max_depth: usize,
    time_limit_ms: Option<u64>,
    nnue: &NNUEModel,
    tt: &mut TranspositionTableAB,
    stop_signal: Option<Arc<AtomicBool>>,
    mut on_progress: F,
) -> SearchResult
where
    F: FnMut(&SearchProgress),
{
    let mut state = SearchState::new(nnue, tt, stop_signal);
    state.time_limit_ms = time_limit_ms;

    // Nạp lịch sử ván cờ vào để kiểm tra lặp cờ
    for h in history {
        state.game_history.push(h.hash);
    }

    // Khởi tạo Accumulator tầng 0 từ FEN hiện tại
    state.acc_stack[0].refresh_full(board, nnue);
    state.tt.new_search();

    let mut best_move = None;
    let mut best_score: i16 = 0;
    let mut last_completed_depth = 0;

    let start_time = Instant::now();

    for current_depth in 1..=max_depth {
        let mut alpha = -VALUE_INFINITE;
        let mut beta = VALUE_INFINITE;

        // Aspiration Windows: Thu hẹp cửa sổ tìm kiếm quanh điểm số của độ sâu trước
        if current_depth >= 5 && best_score.abs() < VALUE_MATE_THRESHOLD {
            let delta = 25;
            alpha = (best_score - delta).max(-VALUE_INFINITE);
            beta = (best_score + delta).min(VALUE_INFINITE);
        }

        let mut score;

        loop {
            score = negamax_pvs(
                board,
                0,
                current_depth as i8,
                alpha,
                beta,
                false,
                &mut state,
            );

            if state.stopped {
                break;
            }

            // Nếu trượt khỏi cửa sổ Aspiration, mở rộng cửa sổ và tìm lại
            if score <= alpha && alpha > -VALUE_INFINITE {
                alpha = -VALUE_INFINITE;
            } else if score >= beta && beta < VALUE_INFINITE {
                beta = VALUE_INFINITE;
            } else {
                break;
            }
        }

        if state.stopped {
            break;
        }

        best_score = score;
        last_completed_depth = current_depth;

        // Lấy best move từ TT
        let (root_tt_move, _) = state.tt.probe(board.zobrist_key, 0, current_depth as i8, -VALUE_INFINITE, VALUE_INFINITE);
        if let Some(m) = root_tt_move {
            best_move = Some(m);
        }

        let elapsed_ms = start_time.elapsed().as_millis().max(1) as u64;
        let nps = (state.nodes * 1000) / elapsed_ms;
        let pv_str = extract_pv_string(board, state.tt, current_depth.min(10));

        let progress = SearchProgress {
            depth: current_depth,
            score: best_score,
            time_ms: elapsed_ms,
            nodes: state.nodes,
            nps,
            pv: pv_str,
            best_move,
        };
        on_progress(&progress);

        // Dừng sớm nếu đã tìm thấy nước chiếu bí sát cục
        if best_score.abs() > VALUE_MATE_THRESHOLD {
            break;
        }
    }

    let elapsed_ms = start_time.elapsed().as_millis().max(1) as u64;
    let nps = (state.nodes * 1000) / elapsed_ms;

    SearchResult {
        best_move,
        score: best_score,
        depth: last_completed_depth,
        nodes: state.nodes,
        time_ms: elapsed_ms,
        nps,
    }
}

pub fn search_best_move(
    board: &mut Board,
    history: &[HistoryEntry],
    max_depth: usize,
    time_limit_ms: Option<u64>,
    nnue: &NNUEModel,
    tt: &mut TranspositionTableAB,
    stop_signal: Option<Arc<AtomicBool>>,
    verbose: bool,
) -> SearchResult {
    search_best_move_with_callback(
        board,
        history,
        max_depth,
        time_limit_ms,
        nnue,
        tt,
        stop_signal,
        |prog| {
            if verbose {
                println!(
                    "info depth {} score cp {} time {} nodes {} nps {} pv {}",
                    prog.depth, prog.score, prog.time_ms, prog.nodes, prog.nps, prog.pv
                );
            }
        },
    )
}

/// Tìm kiếm danh sách Top-N nước đi có điểm số đánh giá cao nhất ở vị trí hiện tại
pub fn search_top_moves(
    board: &mut Board,
    history: &[HistoryEntry],
    max_depth: usize,
    top_n: usize,
    nnue: &NNUEModel,
    tt: &mut TranspositionTableAB,
) -> Vec<(Move, i16)> {
    if top_n <= 1 {
        let res = search_best_move(board, history, max_depth, None, nnue, tt, None, false);
        if let Some(m) = res.best_move {
            return vec![(m, res.score)];
        } else {
            return Vec::new();
        }
    }

    let res = search_best_move(board, history, max_depth, None, nnue, tt, None, false);
    let best_move = match res.best_move {
        Some(m) => m,
        None => return Vec::new(),
    };

    let all_moves = board.generate_all_moves();
    let moving_side = board.side_to_move;
    let mut legal_moves: Vec<Move> = Vec::new();
    for m in all_moves {
        let undo = board.make_move(m);
        if !board.kings_facing() && !board.is_in_check(moving_side) {
            legal_moves.push(m);
        }
        board.unmake_move(m, undo);
    }

    if legal_moves.len() <= 1 {
        return vec![(best_move, res.score)];
    }

    let mut scored_moves: Vec<(Move, i16)> = Vec::with_capacity(legal_moves.len());
    scored_moves.push((best_move, res.score));

    let mut state = SearchState::new(nnue, tt, None);
    for h in history {
        state.game_history.push(h.hash);
    }
    state.acc_stack[0].refresh_full(board, nnue);

    let shallow_depth = (max_depth.saturating_sub(1)).max(1) as i8;

    for &m in &legal_moves {
        if m == best_move {
            continue;
        }

        let from_sq = m.from_sq();
        let to_sq = m.to_sq();
        let captured = board.piece_at(to_sq);
        let moving_piece = board.piece_at(from_sq).unwrap();

        let undo = board.make_move(m);

        let (_tt_move, tt_val) = state.tt.probe(board.zobrist_key, 1, shallow_depth, -VALUE_INFINITE, VALUE_INFINITE);
        let score = if let Some(val) = tt_val {
            -val
        } else {
            state.acc_stack[1] = state.acc_stack[0];
            state.acc_stack[1].update_move(board, moving_piece, from_sq, to_sq, captured, nnue);
            state.game_history.push(board.zobrist_key);

            let s = -negamax_pvs(
                board,
                1,
                shallow_depth,
                -VALUE_INFINITE,
                VALUE_INFINITE,
                false,
                &mut state,
            );

            state.game_history.pop();
            s
        };

        board.unmake_move(m, undo);
        scored_moves.push((m, score));
    }

    scored_moves.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    scored_moves.truncate(top_n);
    scored_moves
}

