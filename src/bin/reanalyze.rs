use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crossbeam_channel::Sender;

use wisdom::board::Board;
use wisdom::eval_queue::{EvalQueue, EvalRequest};
use wisdom::mcts::MCTS;
use wisdom::r#move::Move;
use wisdom::nn::XiangqiOnnx;
use wisdom::tt::TranspositionTable;

// =====================================================================
// HÀM RE-ANALYZE LÕI
// =====================================================================
fn reanalyze_data(
    input_csv: &str,
    output_csv: &str,
    eval_tx: &Sender<EvalRequest>,
    simulations: usize,
) {
    // 1. Đọc toàn bộ FEN từ file cũ
    let file = File::open(input_csv).expect("❌ Không mở được file CSV đầu vào.");
    let reader = BufReader::new(file);
    let mut old_fens = Vec::new();

    for line in reader.lines() {
        if let Ok(l) = line {
            // Lấy phần FEN (trước dấu phẩy hoặc khoảng trắng đầu tiên, do data của bác có thể cách nhau bởi space)
            // Lưu ý: Dữ liệu mẫu của bác có dạng: FEN value policy cách nhau bằng dấu cách.
            // Dòng code dưới đây xử lý tự động cả trường hợp cách nhau bằng dấu phẩy hoặc khoảng trắng.
            let parts: Vec<&str> = if l.contains(',') {
                l.split(',').collect()
            } else {
                l.split_whitespace().collect()
            };

            // Một FEN cờ tướng (bao gồm bàn cờ và lượt đi) chiếm 2 phần đầu tiên nếu cách bằng khoảng trắng
            // Ví dụ: "rnbakabnr/9/... w 0.0083 5827" -> Lấy "rnbakabnr/9/... w"
            if parts.len() >= 2 {
                let fen_string = if parts[1] == "w" || parts[1] == "b" {
                    format!("{} {}", parts[0], parts[1])
                } else {
                    parts[0].to_string()
                };
                old_fens.push(fen_string);
            }
        }
    }

    let total_fens = old_fens.len();
    println!(
        "🔄 Đã nạp {} FENs để Re-analyze với {} Simulations...",
        total_fens, simulations
    );

    // Khởi tạo vector chứa kết quả với kích thước đúng bằng tổng số FEN để giữ thứ tự
    let results = Arc::new(Mutex::new(vec![String::new(); total_fens]));
    let fens_iterator = Arc::new(Mutex::new(old_fens.into_iter().enumerate()));

    let start_time = Instant::now();
    let processed_count = Arc::new(Mutex::new(0));

    // 🎯 SỬ DỤNG CHUNG 1 BẢNG HASH KHỔNG LỒ (1024 MB) ĐỂ TĂNG TỐC KHI RE-ANALYZE
    let shared_tt = Arc::new(TranspositionTable::new(16384));

    // 2. Chạy 128 Luồng băm nát đống FEN này
    std::thread::scope(|s| {
        for _ in 0..512 {
            // Giữ CPU bận rộn
            let iterator_clone = Arc::clone(&fens_iterator);
            let results_clone = Arc::clone(&results);
            let count_clone = Arc::clone(&processed_count);
            let tx = eval_tx.clone();

            // Clone tham chiếu Hash Table cho luồng này dùng chung
            let local_tt = Arc::clone(&shared_tt);

            s.spawn(move || {
                let mcts = MCTS::new(500_000);

                loop {
                    let item = {
                        let mut it = iterator_clone.lock().unwrap();
                        it.next()
                    };

                    match item {
                        Some((idx, fen)) => {
                            if let Ok(board) = Board::from_fen(&fen) {
                                // Gọi MCTS với add_noise = true
                                let (best_move, metrics) = mcts.search_best_move(
                                    &board,
                                    &[], // History rỗng vì phân tích tĩnh
                                    simulations,
                                    &tx,
                                    &local_tt,
                                    1,
                                    true, // Bật Dirichlet Noise
                                );

                                // Xác định quân nào đang đi để tính Index 8100
                                let is_black = board.side_to_move == wisdom::board::Color::Black;
                                let best_policy_idx = Move::move_to_nn_index(best_move, is_black);

                                // Dùng thẳng Q-value từ root (chính là metrics.eval)
                                let new_value = metrics.eval;

                                // Lưu kết quả vào RAM tại đúng vị trí ban đầu
                                {
                                    let mut res = results_clone.lock().unwrap();
                                    res[idx] = format!("{},{},{}", fen, new_value, best_policy_idx);
                                }

                                // 🎯 IN DẤU CHẤM BÁO HIỆU XONG 1 FEN
                                print!(".");
                                let _ = std::io::stdout().flush();

                                // In tiến độ
                                let mut c = count_clone.lock().unwrap();
                                *c += 1;
                                if *c % 1000 == 0 || *c == total_fens {
                                    println!("\n⏳ Đã Re-analyze {}/{} vị trí...", *c, total_fens);
                                }
                            }
                        }
                        None => break,
                    }
                }
            });
        }
    });

    println!(
        "\n💾 Đang ghi toàn bộ kết quả ra đĩa cứng tại {}...",
        output_csv
    );
    let output_file = File::create(output_csv).expect("❌ Không tạo được file CSV đầu ra.");
    let mut file_writer = BufWriter::new(output_file);
    let all_results = results.lock().unwrap();
    let mut written_count = 0;
    for r in all_results.iter() {
        if !r.is_empty() {
            writeln!(file_writer, "{}", r).unwrap();
            written_count += 1;
        }
    }

    println!(
        "✅ Xong! Hoàn thành Re-analyze và lưu {} FENs mất {:.2?}",
        written_count,
        start_time.elapsed()
    );
}

// =====================================================================
// HÀM MAIN
// =====================================================================
fn main() {
    // 1. Nhận tham số từ dòng lệnh
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "⚠️ Cách dùng: cargo run --bin reanalyze --release <FILE_CSV_DAU_VAO> <MODEL_ONNX_PATH>"
        );
        eprintln!("Ví dụ: cargo run --bin reanalyze --release data_cu.csv wisdom_net_v0.onnx");
        std::process::exit(1);
    }

    let input_csv = &args[1];
    let model_path = &args[2];
    let simulations = 1200;

    // Tự động tạo tên file output bằng cách bỏ đuôi .csv và thêm _reanalyzed.csv
    let output_csv = if input_csv.ends_with(".csv") {
        input_csv.replace(".csv", "_reanalyzed.csv")
    } else {
        format!("{}_reanalyzed.csv", input_csv)
    };

    println!("============================================================");
    println!("🚀 KHỞI ĐỘNG CÔNG CỤ RE-ANALYZE");
    println!("📂 File đầu vào  : {}", input_csv);
    println!("💾 File đầu ra   : {}", output_csv);
    println!("🧠 Model ONNX    : {}", model_path);
    println!("⚙️  Simulations   : {}", simulations);
    println!("============================================================\n");

    // 2. Khởi tạo Model và EvalQueue
    println!("📥 Đang nạp Model từ '{}' vào GPU...", model_path);
    let onnx_model = XiangqiOnnx::new(model_path);
    let eval_queue = EvalQueue::new(onnx_model, 512, 100);

    // 3. Tiến hành Re-analyze
    reanalyze_data(input_csv, &output_csv, &eval_queue.tx, simulations);
}
