use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use crate::board::{Board, Color, Piece, PieceType};

pub const L1_DIM: usize = 512;
pub const L2_DIM: usize = 32;
pub const L3_DIM: usize = 32;
pub const NUM_FEATURES: usize = 11340;

pub const QA: f32 = 255.0;
pub const QB: f32 = 64.0;

// =====================================================================
// 1. ACCUMULATOR (Bộ tích lũy vi phân cho 2 góc nhìn)
// =====================================================================
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct Accumulator {
    // [0]: Góc nhìn Đỏ, [1]: Góc nhìn Đen
    pub accumulation: [[i16; L1_DIM]; 2],
}

impl Accumulator {
    pub fn new(biases: &[i16; L1_DIM]) -> Self {
        Self {
            accumulation: [*biases, *biases],
        }
    }

    #[inline(always)]
    pub fn add_feature(&mut self, perspective_idx: usize, feature_idx: usize, weights: &[i16]) {
        let offset = feature_idx * L1_DIM;
        let w = &weights[offset..offset + L1_DIM];
        let acc = &mut self.accumulation[perspective_idx];
        for i in 0..L1_DIM {
            acc[i] += w[i];
        }
    }

    #[inline(always)]
    pub fn sub_feature(&mut self, perspective_idx: usize, feature_idx: usize, weights: &[i16]) {
        let offset = feature_idx * L1_DIM;
        let w = &weights[offset..offset + L1_DIM];
        let acc = &mut self.accumulation[perspective_idx];
        for i in 0..L1_DIM {
            acc[i] -= w[i];
        }
    }

    /// Tính lại toàn bộ Accumulator từ đầu (Scratch Refresh)
    pub fn refresh_full(&mut self, board: &Board, model: &NNUEModel) {
        self.accumulation[0] = model.ft_bias;
        self.accumulation[1] = model.ft_bias;

        let kr_idx = model.get_red_king_idx(board);
        let kb_idx = model.get_black_king_idx(board);

        for sq in 0..256 {
            if let Some(piece) = board.piece_at(sq) {
                let dense_sq = Board::square_to_dense(sq);

                // Góc nhìn Đỏ
                let p_red = model.piece_to_index(piece, Color::Red);
                let f_red = kr_idx * 1260 + p_red * 90 + dense_sq;
                self.add_feature(0, f_red, &model.ft_weights);

                // Góc nhìn Đen (xoay 180 độ: sq_rot = 89 - dense_sq)
                let p_black = model.piece_to_index(piece, Color::Black);
                let f_black = kb_idx * 1260 + p_black * 90 + (89 - dense_sq);
                self.add_feature(1, f_black, &model.ft_weights);
            }
        }
    }

    /// Làm mới 1 góc nhìn khi Tướng của bên đó di chuyển
    pub fn refresh_side(&mut self, perspective_idx: usize, board: &Board, model: &NNUEModel) {
        self.accumulation[perspective_idx] = model.ft_bias;

        if perspective_idx == 0 {
            // Đỏ
            let kr_idx = model.get_red_king_idx(board);
            for sq in 0..256 {
                if let Some(piece) = board.piece_at(sq) {
                    let dense_sq = Board::square_to_dense(sq);
                    let p_red = model.piece_to_index(piece, Color::Red);
                    let f_red = kr_idx * 1260 + p_red * 90 + dense_sq;
                    self.add_feature(0, f_red, &model.ft_weights);
                }
            }
        } else {
            // Đen
            let kb_idx = model.get_black_king_idx(board);
            for sq in 0..256 {
                if let Some(piece) = board.piece_at(sq) {
                    let dense_sq = Board::square_to_dense(sq);
                    let p_black = model.piece_to_index(piece, Color::Black);
                    let f_black = kb_idx * 1260 + p_black * 90 + (89 - dense_sq);
                    self.add_feature(1, f_black, &model.ft_weights);
                }
            }
        }
    }

    /// Cập nhật vi phân O(1) sau khi đi nước cờ Move
    pub fn update_move(
        &mut self,
        board_after: &Board,
        moving_piece: Piece,
        from_sq: usize,
        to_sq: usize,
        captured: Option<Piece>,
        model: &NNUEModel,
    ) {
        let from_dense = Board::square_to_dense(from_sq);
        let to_dense = Board::square_to_dense(to_sq);

        // Trường hợp 1: Tướng Đỏ đi
        if moving_piece.piece_type == PieceType::King && moving_piece.color == Color::Red {
            self.refresh_side(0, board_after, model);

            // Cập nhật vi phân cho góc nhìn Đen
            let kb_idx = model.get_black_king_idx(board_after);
            let p_black = model.piece_to_index(moving_piece, Color::Black);
            self.sub_feature(1, kb_idx * 1260 + p_black * 90 + (89 - from_dense), &model.ft_weights);
            self.add_feature(1, kb_idx * 1260 + p_black * 90 + (89 - to_dense), &model.ft_weights);
            if let Some(cap) = captured {
                let cap_black = model.piece_to_index(cap, Color::Black);
                self.sub_feature(1, kb_idx * 1260 + cap_black * 90 + (89 - to_dense), &model.ft_weights);
            }
            return;
        }

        // Trường hợp 2: Tướng Đen đi
        if moving_piece.piece_type == PieceType::King && moving_piece.color == Color::Black {
            self.refresh_side(1, board_after, model);

            // Cập nhật vi phân cho góc nhìn Đỏ
            let kr_idx = model.get_red_king_idx(board_after);
            let p_red = model.piece_to_index(moving_piece, Color::Red);
            self.sub_feature(0, kr_idx * 1260 + p_red * 90 + from_dense, &model.ft_weights);
            self.add_feature(0, kr_idx * 1260 + p_red * 90 + to_dense, &model.ft_weights);
            if let Some(cap) = captured {
                let cap_red = model.piece_to_index(cap, Color::Red);
                self.sub_feature(0, kr_idx * 1260 + cap_red * 90 + to_dense, &model.ft_weights);
            }
            return;
        }

        // Trường hợp 3: Quân thường đi (Cập nhật vi phân cho cả 2 bên - O(1) Siêu tốc!)
        let kr_idx = model.get_red_king_idx(board_after);
        let kb_idx = model.get_black_king_idx(board_after);

        // Góc nhìn Đỏ
        let p_red = model.piece_to_index(moving_piece, Color::Red);
        self.sub_feature(0, kr_idx * 1260 + p_red * 90 + from_dense, &model.ft_weights);
        self.add_feature(0, kr_idx * 1260 + p_red * 90 + to_dense, &model.ft_weights);
        if let Some(cap) = captured {
            let cap_red = model.piece_to_index(cap, Color::Red);
            self.sub_feature(0, kr_idx * 1260 + cap_red * 90 + to_dense, &model.ft_weights);
        }

        // Góc nhìn Đen
        let p_black = model.piece_to_index(moving_piece, Color::Black);
        self.sub_feature(1, kb_idx * 1260 + p_black * 90 + (89 - from_dense), &model.ft_weights);
        self.add_feature(1, kb_idx * 1260 + p_black * 90 + (89 - to_dense), &model.ft_weights);
        if let Some(cap) = captured {
            let cap_black = model.piece_to_index(cap, Color::Black);
            self.sub_feature(1, kb_idx * 1260 + cap_black * 90 + (89 - to_dense), &model.ft_weights);
        }
    }
}

// =====================================================================
// 2. MÔ HÌNH NNUE (LOAD TỪ FILE NHỊ PHÂN .BIN)
// =====================================================================
pub struct NNUEModel {
    pub ft_bias: [i16; L1_DIM],
    pub ft_weights: Vec<i16>, // 11,340 * 256

    pub fc0_bias: [i32; L2_DIM],
    pub fc0_weights: Vec<i8>, // 32 * 512

    pub fc1_bias: [i32; L3_DIM],
    pub fc1_weights: Vec<i8>, // 32 * 32

    pub fc2_bias: f32,
    pub fc2_weights: [f32; L3_DIM],
    /// Cờ bật chế độ Material Bootstrap (dùng hàm lượng giá vật chất board.evaluate() thay vì NNUE)
    pub use_material: bool,
}

impl NNUEModel {
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let file = File::open(&path).map_err(|e| format!("Không thể mở file: {}", e))?;
        let mut reader = BufReader::new(file);

        // 1. Đọc Header
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic).map_err(|e| e.to_string())?;
        if &magic != b"WSDMNNUE" {
            return Err("File NNUE không hợp lệ (Sai magic header)".to_string());
        }

        let mut buf_u32 = [0u8; 4];
        reader.read_exact(&mut buf_u32).map_err(|e| e.to_string())?;
        let _version = u32::from_le_bytes(buf_u32);

        reader.read_exact(&mut buf_u32).map_err(|e| e.to_string())?;
        let l1 = u32::from_le_bytes(buf_u32) as usize;

        reader.read_exact(&mut buf_u32).map_err(|e| e.to_string())?;
        let l2 = u32::from_le_bytes(buf_u32) as usize;

        reader.read_exact(&mut buf_u32).map_err(|e| e.to_string())?;
        let l3 = u32::from_le_bytes(buf_u32) as usize;

        reader.read_exact(&mut buf_u32).map_err(|e| e.to_string())?;
        let num_feat = u32::from_le_bytes(buf_u32) as usize;

        if l1 != L1_DIM || l2 != L2_DIM || l3 != L3_DIM || num_feat != NUM_FEATURES {
            return Err(format!(
                "Kích thước mạng không khớp: L1={}, L2={}, L3={}, Feat={}",
                l1, l2, l3, num_feat
            ));
        }

        // 2. Đọc Feature Transformer Biases [i16; 256]
        let mut ft_bias = [0i16; L1_DIM];
        let mut buf_i16 = [0u8; 2];
        for b in ft_bias.iter_mut() {
            reader.read_exact(&mut buf_i16).map_err(|e| e.to_string())?;
            *b = i16::from_le_bytes(buf_i16);
        }

        // 3. Đọc Feature Transformer Weights [i16; 11340 * 256]
        let total_ft_weights = NUM_FEATURES * L1_DIM;
        let mut ft_weights = vec![0i16; total_ft_weights];
        let mut bytes_buf = vec![0u8; total_ft_weights * 2];
        reader.read_exact(&mut bytes_buf).map_err(|e| e.to_string())?;
        for (i, chunk) in bytes_buf.chunks_exact(2).enumerate() {
            ft_weights[i] = i16::from_le_bytes([chunk[0], chunk[1]]);
        }

        // 4. Đọc FC0 Biases [i32; 32]
        let mut fc0_bias = [0i32; L2_DIM];
        for b in fc0_bias.iter_mut() {
            reader.read_exact(&mut buf_u32).map_err(|e| e.to_string())?;
            *b = i32::from_le_bytes(buf_u32);
        }

        // 5. Đọc FC0 Weights [i8; 32 * 512]
        let total_fc0_weights = L2_DIM * (L1_DIM * 2);
        let mut fc0_weights = vec![0i8; total_fc0_weights];
        let mut bytes_fc0 = vec![0u8; total_fc0_weights];
        reader.read_exact(&mut bytes_fc0).map_err(|e| e.to_string())?;
        for (i, &byte) in bytes_fc0.iter().enumerate() {
            fc0_weights[i] = byte as i8;
        }

        // 6. Đọc FC1 Biases [i32; 32]
        let mut fc1_bias = [0i32; L3_DIM];
        for b in fc1_bias.iter_mut() {
            reader.read_exact(&mut buf_u32).map_err(|e| e.to_string())?;
            *b = i32::from_le_bytes(buf_u32);
        }

        // 7. Đọc FC1 Weights [i8; 32 * 32]
        let total_fc1_weights = L3_DIM * L2_DIM;
        let mut fc1_weights = vec![0i8; total_fc1_weights];
        let mut bytes_fc1 = vec![0u8; total_fc1_weights];
        reader.read_exact(&mut bytes_fc1).map_err(|e| e.to_string())?;
        for (i, &byte) in bytes_fc1.iter().enumerate() {
            fc1_weights[i] = byte as i8;
        }

        // 8. Đọc FC2 Bias (f32)
        reader.read_exact(&mut buf_u32).map_err(|e| e.to_string())?;
        let fc2_bias = f32::from_le_bytes(buf_u32);

        // 9. Đọc FC2 Weights [f32; 32]
        let mut fc2_weights = [0f32; L3_DIM];
        for w in fc2_weights.iter_mut() {
            reader.read_exact(&mut buf_u32).map_err(|e| e.to_string())?;
            *w = f32::from_le_bytes(buf_u32);
        }

        Ok(Self {
            ft_bias,
            ft_weights,
            fc0_bias,
            fc0_weights,
            fc1_bias,
            fc1_weights,
            fc2_bias,
            fc2_weights,
            use_material: false,
        })
    }

    /// Khởi tạo model chỉ dùng hàm Material Evaluation (dùng cho Thế hệ 0 - Bootstrap)
    pub fn new_material_only() -> Self {
        Self {
            ft_bias: [0i16; L1_DIM],
            ft_weights: vec![0i16; NUM_FEATURES * L1_DIM],
            fc0_bias: [0i32; L2_DIM],
            fc0_weights: vec![0i8; L2_DIM * (L1_DIM * 2)],
            fc1_bias: [0i32; L3_DIM],
            fc1_weights: vec![0i8; L3_DIM * L2_DIM],
            fc2_bias: 0.0,
            fc2_weights: [0.0; L3_DIM],
            use_material: true,
        }
    }

    #[inline(always)]
    pub fn get_red_king_idx(&self, board: &Board) -> usize {
        let r = (board.red_king_sq >> 4).clamp(7, 9);
        let c = (board.red_king_sq & 15).clamp(3, 5);
        (r - 7) * 3 + (c - 3)
    }

    #[inline(always)]
    pub fn get_black_king_idx(&self, board: &Board) -> usize {
        let r_orig = (board.black_king_sq >> 4).clamp(0, 2);
        let c_orig = (board.black_king_sq & 15).clamp(3, 5);
        let rot_r = (9 - r_orig).clamp(7, 9);
        let rot_c = (8 - c_orig).clamp(3, 5);
        (rot_r - 7) * 3 + (rot_c - 3)
    }

    #[inline(always)]
    pub fn piece_to_index(&self, piece: Piece, perspective: Color) -> usize {
        let type_offset = match piece.piece_type {
            PieceType::King => 0,
            PieceType::Advisor => 1,
            PieceType::Elephant => 2,
            PieceType::Horse => 3,
            PieceType::Rook => 4,
            PieceType::Cannon => 5,
            PieceType::Pawn => 6,
        };
        if piece.color == perspective {
            type_offset
        } else {
            type_offset + 7
        }
    }

    /// Suy luận NNUE (Forward Pass) siêu tốc
    /// Trả về điểm thẩm định trong khoảng [-1.0, 1.0] theo góc nhìn của bên đang đi (Side To Move)
    #[inline(always)]
    pub fn evaluate(&self, board: &Board, acc: &Accumulator) -> f32 {
        let (us_idx, them_idx) = if board.side_to_move == Color::Red {
            (0, 1)
        } else {
            (1, 0)
        };

        // 1. Activation ClippedReLU cho Accumulator: kẹp vào khoảng [0, 255]
        let mut fc0_in = [0i32; L1_DIM * 2];
        let acc_us = &acc.accumulation[us_idx];
        let acc_them = &acc.accumulation[them_idx];

        for i in 0..L1_DIM {
            fc0_in[i] = acc_us[i].clamp(0, 255) as i32;
            fc0_in[i + L1_DIM] = acc_them[i].clamp(0, 255) as i32;
        }

        // 2. Hidden Layer 0: 512 -> 32
        const QA_QB: f32 = QA * QB; // 255.0 * 64.0 = 16320.0
        let mut act0 = [0.0f32; L2_DIM];
        for i in 0..L2_DIM {
            let offset = i * (L1_DIM * 2);
            let w = &self.fc0_weights[offset..offset + (L1_DIM * 2)];
            let mut dot = self.fc0_bias[i];
            for j in 0..(L1_DIM * 2) {
                dot += fc0_in[j] * (w[j] as i32);
            }
            act0[i] = ((dot as f32) / QA_QB).clamp(0.0, 1.0);
        }

        // 3. Hidden Layer 1: 32 -> 32
        const QB_QB: f32 = QB * QB; // 64.0 * 64.0 = 4096.0
        let mut act0_i8 = [0i32; L2_DIM];
        for i in 0..L2_DIM {
            act0_i8[i] = (act0[i] * QB).round() as i32;
        }

        let mut act1 = [0.0f32; L3_DIM];
        for i in 0..L3_DIM {
            let offset = i * L2_DIM;
            let w = &self.fc1_weights[offset..offset + L2_DIM];
            let mut dot = self.fc1_bias[i];
            for j in 0..L2_DIM {
                dot += act0_i8[j] * (w[j] as i32);
            }
            act1[i] = ((dot as f32) / QB_QB).clamp(0.0, 1.0);
        }

        // 4. Output Layer 2: 32 -> 1
        let mut out = self.fc2_bias;
        for i in 0..L3_DIM {
            out += act1[i] * self.fc2_weights[i];
        }

        out.tanh()
    }

    /// Thẩm định điểm theo đơn vị Centipawns (-1000 đến +1000)
    #[inline(always)]
    pub fn evaluate_cp(&self, board: &Board, acc: &Accumulator) -> i16 {
        if self.use_material {
            return board.evaluate().clamp(-1000, 1000) as i16;
        }
        (self.evaluate(board, acc) * 1000.0).clamp(-1000.0, 1000.0) as i16
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;

    #[test]
    fn test_nnue_scratch_vs_incremental() {
        let model_path = "./wisdom_models/wisdom_nnue.bin";
        if !std::path::Path::new(model_path).exists() {
            return;
        }
        let model = match NNUEModel::load_from_file(model_path) {
            Ok(m) => m,
            Err(_) => return, // Bỏ qua test nếu file model cũ khác kích thước L1_DIM
        };
        let mut board = Board::new();
        board.set_initial_position();

        let mut acc = Accumulator::new(&model.ft_bias);
        acc.refresh_full(&board, &model);

        let mut moves = board.generate_captures();
        moves.append(&mut board.generate_quiets());
        let m = moves[0];
        let p = board.piece_at(m.from_sq()).unwrap();
        let cap = board.piece_at(m.to_sq());
        board.make_move(m);

        let mut inc_acc = acc;
        inc_acc.update_move(&board, p, m.from_sq(), m.to_sq(), cap, &model);

        let mut scratch = Accumulator::new(&model.ft_bias);
        scratch.refresh_full(&board, &model);

        assert_eq!(inc_acc.accumulation, scratch.accumulation);
    }
}

