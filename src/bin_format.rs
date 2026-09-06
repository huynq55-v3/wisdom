use crate::board::{Board, Color, Piece, PieceType};

/// WisdomBinRecord: Định dạng nhị phân lưu trữ thế cờ siêu gọn (96 bytes).
/// 96 bytes là bội số của 16 bytes (khớp hoàn hảo CPU cache line).
#[repr(C, packed)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WisdomBinRecord {
    /// 90 ô bàn cờ (10 hàng x 9 cột):
    /// 255 (0xFF): Ô trống
    /// 0..6: Quân Đỏ (Tướng=0, Sĩ=1, Tượng=2, Mã=3, Xe=4, Pháo=5, Tốt=6)
    /// 7..13: Quân Đen (Tướng=7, Sĩ=8, Tượng=9, Mã=10, Xe=11, Pháo=12, Tốt=13)
    pub squares: [u8; 90],
    /// Lượt đi: 0 = Đỏ (Red), 1 = Đen (Black)
    pub side_to_move: u8,
    /// Điểm thẩm định Alpha-Beta theo góc nhìn của bên đi (centipawns: -30000..30000)
    pub score: i16,
    /// Kết quả chung cuộc theo góc nhìn của bên đi: 1 = Thắng, 0 = Hòa, -1 = Thua
    pub outcome: i8,
    /// Số nước đã đi của ván cờ (ply)
    pub ply: u16,
}

pub const RECORD_SIZE: usize = std::mem::size_of::<WisdomBinRecord>();
const _: () = assert!(RECORD_SIZE == 96);

impl WisdomBinRecord {
    pub fn new(board: &Board, score: i16, outcome: i8, ply: u16) -> Self {
        let mut squares = [255u8; 90];

        for row in 0..10 {
            for col in 0..9 {
                let sq256 = Board::coord_to_square(row, col);
                let sq90 = row * 9 + col;
                if let Some(p) = board.piece_at(sq256) {
                    squares[sq90] = piece_to_code(p);
                }
            }
        }

        let stm = match board.side_to_move {
            Color::Red => 0,
            Color::Black => 1,
        };

        Self {
            squares,
            side_to_move: stm,
            score,
            outcome,
            ply,
        }
    }

    pub fn as_bytes(&self) -> &[u8; 96] {
        unsafe { &*(self as *const Self as *const [u8; 96]) }
    }

    pub fn from_bytes(bytes: &[u8; 96]) -> Self {
        unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const Self) }
    }
}

pub fn piece_to_code(p: Piece) -> u8 {
    let base = match p.color {
        Color::Red => 0,
        Color::Black => 7,
    };
    let type_offset = match p.piece_type {
        PieceType::King => 0,
        PieceType::Advisor => 1,
        PieceType::Elephant => 2,
        PieceType::Horse => 3,
        PieceType::Rook => 4,
        PieceType::Cannon => 5,
        PieceType::Pawn => 6,
    };
    base + type_offset
}
