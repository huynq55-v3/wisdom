use wisdom::ucci::ucci_loop;

const MODEL_PATH: &str = "./wisdom_models/wisdom_net_v0.onnx";

fn start_engine() {
    println!("📦 Khởi tạo ONNX model từ: {}", MODEL_PATH);
    let model = wisdom::nn::XiangqiOnnx::new(MODEL_PATH);
    ucci_loop(model);
}

fn main() {
    println!("🚀 Khởi động Wisdom Engine (MCTS + ONNX)...");
    start_engine();
}
