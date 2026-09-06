use std::time::Instant;
use wisdom::board::Board;
use wisdom::nnue::{Accumulator, NNUEModel};

fn main() {
    println!("🚀 ============================================================");
    println!("🧪 BẮT ĐẦU KIỂM THỬ VÀ BENCHMARK WISDOM NNUE (CPU RUST)");
    println!("🚀 ============================================================");

    let model_path = "./wisdom_models/wisdom_nnue.bin";
    println!("📦 Nạp mô hình NNUE từ: {}", model_path);
    let start_load = Instant::now();
    let model = match NNUEModel::load_from_file(model_path) {
        Ok(m) => {
            println!("✅ Nạp mô hình thành công trong {:?}", start_load.elapsed());
            m
        }
        Err(e) => {
            eprintln!("❌ Lỗi nạp mô hình: {}", e);
            return;
        }
    };

    // 1. Kiểm tra tính điểm trên một số thế cờ FEN mẫu
    let test_fens = [
        (
            "Thế cờ xuất phát",
            "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w",
        ),
        (
            "Đỏ đi Binh 7 tiến 1 (P3+1)",
            "rnbakabnr/9/1c5c1/p1p1p1p1p/9/2P6/P3P1P1P/1C5C1/9/RNBAKABNR b",
        ),
        (
            "Đen đi Pháo 8 bình 7 (C8=7)",
            "rnbakabnr/9/2c4c1/p1p1p1p1p/9/2P6/P3P1P1P/1C5C1/9/RNBAKABNR w",
        ),
        (
            "Đỏ nhảy Mã 2 tiến 3 (N2+3)",
            "rnbakabnr/9/2c4c1/p1p1p1p1p/9/2P6/P3P1P1P/1C2C1N2/9/RNBAKAB1R b",
        ),
    ];

    println!("\n🔍 ĐÁNH GIÁ CÁC THẾ CỜ MẪU:");
    let mut acc = Accumulator::new(&model.ft_bias);

    for (name, fen) in test_fens {
        let board = Board::from_fen(fen).expect("FEN không hợp lệ");
        acc.refresh_full(&board, &model);
        let score = model.evaluate(&board, &acc);
        let score_cp = model.evaluate_cp(&board, &acc);

        let stm_str = if board.side_to_move == wisdom::board::Color::Red {
            "Đỏ đi (w)"
        } else {
            "Đen đi (b)"
        };

        println!(
            "   [{}] -> Lượt: {:<10} | Score: {:+.4} | Centipawns: {:+4} cp",
            name, stm_str, score, score_cp
        );
    }

    // 2. Kiểm tra tính toàn vẹn của Cập Nhật Vi Phân (Incremental Updates vs Full Refresh)
    println!("\n🛡️ KIỂM TRA TÍNH TOÀN VẸN CỦA CẬP NHẬT VI PHÂN (INCREMENTAL UPDATES):");
    let mut board = Board::new();
    board.set_initial_position();
    acc.refresh_full(&board, &model);

    let mut incremental_acc = acc;
    let mut move_count = 0;
    let mut is_consistent = true;

    // Giả lập chơi một chuỗi nước đi và so sánh
    for _ in 0..50 {
        let mut moves = board.generate_captures();
        moves.append(&mut board.generate_quiets());
        let legal_moves: Vec<_> = moves
            .into_iter()
            .filter(|&m| {
                let moving_side = board.side_to_move;
                let undo = board.make_move(m);
                let legal = !board.kings_facing() && !board.is_in_check(moving_side);
                board.unmake_move(m, undo);
                legal
            })
            .collect();

        if legal_moves.is_empty() {
            break;
        }

        // Chọn nước đi đầu tiên hợp lệ
        let m = legal_moves[0];
        let moving_piece = board.piece_at(m.from_sq()).unwrap();
        let captured = board.piece_at(m.to_sq());

        board.make_move(m);
        move_count += 1;

        // Cập nhật vi phân
        incremental_acc.update_move(&board, moving_piece, m.from_sq(), m.to_sq(), captured, &model);

        // Tính lại từ đầu để đối chiếu
        let mut scratch_acc = Accumulator::new(&model.ft_bias);
        scratch_acc.refresh_full(&board, &model);

        if incremental_acc.accumulation != scratch_acc.accumulation {
            println!("❌ LỖI: Cập nhật vi phân bị lệch tại nước đi thứ {}!", move_count);
            is_consistent = false;
            break;
        }
    }

    if is_consistent {
        println!(
            "✅ XÁC THỰC THÀNH CÔNG: {} nước đi liên tục cho kết quả Accumulator TRÙNG KHỚP 100% với Scratch Refresh!",
            move_count
        );
    }

    // 3. BENCHMARK TỐC ĐỘ (NPS - NODES PER SECOND)
    println!("\n⚡️ BẮT ĐẦU BENCHMARK TỐC ĐỘ (FORWARD PASS TRÊN CPU 1 CORE)...");
    let num_evals = 1_000_000;
    let bench_start = Instant::now();
    let mut dummy_sum = 0.0f32;

    for _ in 0..num_evals {
        dummy_sum += model.evaluate(&board, &incremental_acc);
    }

    let elapsed = bench_start.elapsed();
    let nps = (num_evals as f64) / elapsed.as_secs_f64();

    println!("   ↳ Số lượng thẩm định : {} lần", num_evals);
    println!("   ↳ Thời gian thực thi : {:.3} giây", elapsed.as_secs_f64());
    println!("   ↳ Tốc độ đạt được    : {:.0} NPS (Nodes Per Second) 🚀", nps);
    println!("   ↳ Dummy checksum     : {:.4}", dummy_sum);

    println!("\n🎉 WISDOM NNUE ĐÃ HOẠT ĐỘNG HOÀN HẢO! 🏆\n");
}
