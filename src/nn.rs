use ndarray::Array4;
use openvino::{Core, ElementType, Shape, Tensor};

use crate::board::{Board, Color, PieceType};

// ============================================================
// CONSTANTS (Giữ lại để các file khác gọi)
// ============================================================
pub const BOARD_W: usize = 9;
pub const BOARD_H: usize = 10;
pub const NUM_PLANES: usize = 14;
pub const TENSOR_SIZE: usize = NUM_PLANES * BOARD_W * BOARD_H;
pub const ACTION_SPACE: usize = 8100;

// Hàm hỗ trợ chuyển FEN sang mảng 1D (Nếu bác đang dùng trong nn.rs thì giữ lại)
pub fn board_to_tensor(board: &Board) -> [f32; TENSOR_SIZE] {
    let mut data = [0.0f32; TENSOR_SIZE];
    let is_black = board.side_to_move == Color::Black;

    for row in 0..BOARD_H {
        for col in 0..BOARD_W {
            let sq = Board::coord_to_square(row, col);
            if let Some(piece) = board.piece_at(sq) {
                let piece_offset = match piece.piece_type {
                    PieceType::King => 0,
                    PieceType::Advisor => 1,
                    PieceType::Elephant => 2,
                    PieceType::Horse => 3,
                    PieceType::Rook => 4,
                    PieceType::Cannon => 5,
                    PieceType::Pawn => 6,
                };

                let is_mine = piece.color == board.side_to_move;
                let plane = if is_mine {
                    piece_offset
                } else {
                    piece_offset + 7
                };

                let (mapped_row, mapped_col) = if is_black {
                    (9 - row, 8 - col)
                } else {
                    (row, col)
                };

                let idx = plane * (BOARD_H * BOARD_W) + mapped_row * BOARD_W + mapped_col;
                data[idx] = 1.0;
            }
        }
    }

    data
}

// ============================================================
// OPENVINO MODEL DEFINITION
// ============================================================
// Giữ nguyên tên XiangqiOnnx để khỏi phải sửa ở file eval_queue và ucci
pub struct XiangqiOnnx {
    request: openvino::InferRequest,
}

impl XiangqiOnnx {
    pub fn new(model_path: &str) -> Self {
        let mut core = Core::new().expect("Lỗi khởi tạo OpenVINO.");

        // SỬA LỖI 1: Với file .onnx, nó bao gồm cả weights bên trong.
        // Ta truyền chuỗi rỗng "" cho biến weights_path.
        let model = core
            .read_model_from_file(model_path, "")
            .expect("Không thể đọc file ONNX");

        // SỬA LỖI 2: Dùng .into() để convert sang DeviceType
        let mut compiled_model = core
            .compile_model(&model, "GPU".into())
            .expect("Không thể biên dịch model cho GPU Intel");

        let request = compiled_model
            .create_infer_request()
            .expect("Lỗi tạo Infer Request");

        Self { request }
    }

    pub fn forward(&mut self, batch_array: Array4<f32>) -> (Vec<f32>, Vec<f32>) {
        let shape_info = batch_array.shape();
        let batch_size = shape_info[0];
        let data_slice = batch_array.as_slice().unwrap();

        // SỬA LỖI 3: OpenVINO bắt buộc dùng i64 cho Shape, ta cast từ usize sang i64
        let shape = Shape::new(&[
            batch_size as i64,
            NUM_PLANES as i64,
            BOARD_H as i64,
            BOARD_W as i64,
        ])
        .unwrap();

        // SỬA LỖI 4 & 5: Tạo Tensor rỗng trước, sau đó mới copy dữ liệu vào
        let mut tensor = Tensor::new(ElementType::F32, &shape).unwrap();
        let tensor_data: &mut [f32] = tensor.get_data_mut().unwrap();
        tensor_data.copy_from_slice(data_slice);

        // Nạp Tensor vào cổng "input" (Đảm bảo model Python của bác cổng tên là "input")
        self.request.set_tensor("input", &tensor).unwrap();

        // Chạy Suy luận
        self.request.infer().unwrap();

        // Lấy kết quả từ "value" và "policy"
        let value_tensor = self.request.get_tensor("value").unwrap();
        let policy_tensor = self.request.get_tensor("policy").unwrap();

        // SỬA LỖI 6 & 7: Dùng API get_data() thay vì data()
        let value_output: &[f32] = value_tensor.get_data().unwrap();
        let policy_output: &[f32] = policy_tensor.get_data().unwrap();

        (value_output.to_vec(), policy_output.to_vec())
    }
}
