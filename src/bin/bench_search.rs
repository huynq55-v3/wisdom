use std::time::Instant;
use wisdom::ab_tt::TranspositionTableAB;
use wisdom::board::Board;
use wisdom::nnue::NNUEModel;
use wisdom::search::search_best_move;
use wisdom::ucci::parse_fen;

fn main() {
    println!("🚀 ============================================================");
    println!("⚡️ BENCHMARK TÌM KIẾM ALPHA-BETA (PVS/NEGAMAX) WISDOM NNUE");
    println!("🚀 ============================================================");

    let model_path = "./wisdom_models/wisdom_nnue.bin";
    let model = NNUEModel::load_from_file(model_path).expect("Không thể nạp model NNUE");
    let mut tt = TranspositionTableAB::new(64);

    let test_positions = [
        (
            "Thế cờ khai cuộc (Startpos)",
            "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1",
            12, // Depth
        ),
        (
            "Trung cuộc phức tạp (Middlegame)",
            "2bakab2/9/2c1b1nc1/p1p1p1p1p/9/2P6/P3P1P1P/1C2C1N2/4A4/RNBAK1B1R b - - 0 5",
            12, // Depth
        ),
        (
            "Tàn cuộc (Endgame)",
            "4k4/9/9/9/9/9/9/4R4/4A4/4K4 w - - 0 30",
            16, // Depth
        ),
    ];

    let total_start = Instant::now();
    let mut total_nodes = 0u64;

    for (name, fen, depth) in test_positions {
        println!("\n------------------------------------------------------------");
        println!("🎯 Thử nghiệm: {}", name);
        println!("   FEN: {}", fen);

        let mut board = Board::new();
        parse_fen(&mut board, fen);

        tt.clear();
        let start = Instant::now();

        let result = search_best_move(
            &mut board,
            &[],
            depth,
            None,
            &model,
            &mut tt,
            None,
            true, // in info từng depth
        );

        let elapsed = start.elapsed();
        total_nodes += result.nodes;

        let best_move_str = result.best_move.map(|m| wisdom::ucci::move_to_ucci_string(m)).unwrap_or("none".to_string());
        println!("   ↳ Nước cờ tốt nhất (Best Move) : {}", best_move_str);
        println!("   ↳ Độ sâu đạt được (Depth)      : {} plies", result.depth);
        println!("   ↳ Điểm số (Score Centipawns)   : {:+4} cp", result.score);
        println!("   ↳ Tổng số nodes duyệt          : {} nodes", result.nodes);
        println!("   ↳ Thời gian tìm kiếm           : {:.2}s", elapsed.as_secs_f64());
        println!("   ↳ Tốc độ (NPS)                 : {} NPS ⚡️", result.nps);
    }

    let total_elapsed = total_start.elapsed();
    let avg_nps = (total_nodes as f64) / total_elapsed.as_secs_f64();

    println!("\n============================================================");
    println!("🏆 TỔNG KẾT BENCHMARK:");
    println!("   ↳ Tổng số nodes: {} nodes", total_nodes);
    println!("   ↳ Tổng thời gian: {:.2}s", total_elapsed.as_secs_f64());
    println!("   ↳ Tốc độ trung bình: {:.0} NPS", avg_nps);
    println!("============================================================\n");
}
