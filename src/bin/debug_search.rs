use std::time::Instant;
use wisdom::ab_tt::TranspositionTableAB;
use wisdom::board::Board;
use wisdom::nnue::NNUEModel;
use wisdom::search::search_best_move;
use wisdom::ucci::{move_to_ucci_string, parse_fen};

fn main() {
    let model = NNUEModel::load_from_file("./wisdom_models/wisdom_nnue.bin").unwrap();
    let mut tt = TranspositionTableAB::new(64);
    let mut board = Board::new();
    parse_fen(&mut board, "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w");

    println!("Bắt đầu tìm kiếm từ depth 1 đến depth 16...");
    let start = Instant::now();
    let res = search_best_move(&mut board, &[], 16, None, &model, &mut tt, None, true);
    let elapsed = start.elapsed();

    println!("\n✅ KẾT QUẢ TÌM KIẾM:");
    println!("   ↳ Nước cờ tốt nhất (Best Move) : {}", res.best_move.map(move_to_ucci_string).unwrap_or_default());
    println!("   ↳ Độ sâu đạt được (Depth)      : {} plies", res.depth);
    println!("   ↳ Điểm số (Score)              : {:+4} cp", res.score);
    println!("   ↳ Tổng số nodes duyệt          : {} nodes", res.nodes);
    println!("   ↳ Thời gian thực thi           : {:.3}s", elapsed.as_secs_f64());
    println!("   ↳ Tốc độ đạt được              : {} NPS 🚀", res.nps);
}
