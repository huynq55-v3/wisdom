use crate::nn::{ACTION_SPACE, BOARD_H, BOARD_W, NUM_PLANES, TENSOR_SIZE, XiangqiOnnx};
use crossbeam_channel::{Sender, bounded};
use ndarray::Array4;
use std::thread;

pub struct EvalRequest {
    pub tensor_data: [f32; TENSOR_SIZE],
    pub response_tx: Sender<(f32, Option<Vec<f32>>)>,
    pub need_policy: bool,
}

pub struct EvalQueue {
    pub tx: Sender<EvalRequest>,
}

impl EvalQueue {
    /// Spawns a background thread that listens for evaluation requests,
    /// batches them, runs the ONNX forward pass, and sends results back.
    pub fn new(
        mut model: XiangqiOnnx, // 🎯 Đã thêm `mut` ở đây để sở hữu hoàn toàn
        batch_size: usize,
        timeout_micros: u64,
    ) -> Self {
        let (tx, rx) = bounded::<EvalRequest>(1024);

        thread::spawn(move || {
            let mut batch_inputs = Vec::with_capacity(batch_size * TENSOR_SIZE);
            let mut requests: Vec<EvalRequest> = Vec::with_capacity(batch_size);

            // Biến để theo dõi hiệu năng (tùy chọn)
            let mut total_batches = 0u64;

            loop {
                batch_inputs.clear();
                requests.clear();

                // Block until we get the first request
                match rx.recv() {
                    Ok(req) => {
                        batch_inputs.extend_from_slice(&req.tensor_data);
                        requests.push(req);
                    }
                    Err(_) => break, // Channel closed, exit thread
                }

                let start_wait = std::time::Instant::now();

                // Collect more requests up to batch_size with a sliding timeout
                let gap_timeout = std::time::Duration::from_micros(timeout_micros);
                while requests.len() < batch_size {
                    match rx.recv_timeout(gap_timeout) {
                        Ok(req) => {
                            batch_inputs.extend_from_slice(&req.tensor_data);
                            requests.push(req);
                        }
                        Err(_) => break, // Timeout or disconnected
                    }
                }

                let current_batch_size = requests.len();
                if current_batch_size == 0 {
                    continue;
                }

                let wait_duration = start_wait.elapsed();

                // 🎯 DEBUG DÒNG 1: Kiểm tra độ lấp đầy Batch
                if total_batches % 100 == 0 {
                    // Cứ 100 batch in 1 lần cho đỡ rác màn hình
                    println!(
                        "[EvalQueue] Batch size: {}/{} | Chờ gom: {:?} | Hiệu suất lấp đầy: {:.1}%",
                        current_batch_size,
                        batch_size,
                        wait_duration,
                        (current_batch_size as f32 / batch_size as f32) * 100.0
                    );
                }

                let start_gpu = std::time::Instant::now();

                // Chuyển mảng 1D flat thành Array4 của ndarray để nạp vào ONNX
                let batch_array = Array4::from_shape_vec(
                    (current_batch_size, NUM_PLANES, BOARD_H, BOARD_W),
                    batch_inputs.clone(),
                )
                .expect("Kích thước dữ liệu mảng không khớp với Shape (Batch, 14, 10, 9)");

                // Gọi ONNX Model để tính toán (Vì model là mut nên gọi vô tư)
                let (values, policies) = model.forward(batch_array);

                let gpu_duration = start_gpu.elapsed();

                // 🎯 DEBUG DÒNG 2: Kiểm tra tốc độ xử lý của GPU
                if total_batches % 100 == 0 {
                    println!(
                        "[EvalQueue] GPU Latency: {:?} ({:.2} ms/item)",
                        gpu_duration,
                        gpu_duration.as_secs_f64() * 1000.0 / current_batch_size as f64
                    );
                }

                total_batches += 1;

                // Dispatch results to waiting threads
                for (i, req) in requests.drain(..).enumerate() {
                    let v = values[i];
                    let p = if req.need_policy {
                        let start = i * ACTION_SPACE;
                        let end = start + ACTION_SPACE;
                        Some(policies[start..end].to_vec())
                    } else {
                        None
                    };
                    let _ = req.response_tx.send((v, p));
                }
            }
        });

        Self { tx }
    }
}
