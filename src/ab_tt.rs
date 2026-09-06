use crate::r#move::Move;

pub const TT_FLAG_EXACT: u8 = 0;
pub const TT_FLAG_LOWER: u8 = 1; // Beta cutoff (Score >= Beta)
pub const TT_FLAG_UPPER: u8 = 2; // Alpha fail-low (Score <= Alpha)

pub const VALUE_MATE: i16 = 20000;
pub const VALUE_MATE_THRESHOLD: i16 = VALUE_MATE - 256;

#[repr(C, align(16))]
#[derive(Clone, Copy, Default, Debug)]
pub struct TTEntry {
    pub key: u64,       // 8 bytes (Zobrist hash)
    pub best_move: u16, // 2 bytes (Nước đi tối ưu)
    pub score: i16,     // 2 bytes (Điểm số)
    pub depth: i8,      // 1 byte  (Độ sâu)
    pub flag: u8,       // 1 byte  (Exact=0, Lower=1, Upper=2)
    pub age: u16,       // 2 bytes (Thế hệ ván cờ)
}

pub struct TranspositionTableAB {
    pub entries: Vec<TTEntry>,
    mask: usize,
    pub age: u16,
}

impl TranspositionTableAB {
    /// Khởi tạo Transposition Table với dung lượng RAM tính theo Megabytes (MB)
    pub fn new(mb: usize) -> Self {
        let entry_size = std::mem::size_of::<TTEntry>();
        let target_bytes = mb * 1024 * 1024;
        let mut num_entries = target_bytes / entry_size;

        // Làm tròn số lượng entries về lũy thừa của 2 để dùng bitwise AND thay phép chia modulo
        if !num_entries.is_power_of_two() {
            num_entries = num_entries.next_power_of_two() / 2;
        }
        let num_entries = num_entries.max(1024);
        let mask = num_entries - 1;

        Self {
            entries: vec![TTEntry::default(); num_entries],
            mask,
            age: 0,
        }
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        for entry in self.entries.iter_mut() {
            *entry = TTEntry::default();
        }
        self.age = 0;
    }

    #[inline(always)]
    pub fn new_search(&mut self) {
        self.age = self.age.wrapping_add(1);
    }

    #[inline(always)]
    fn score_to_tt(score: i16, ply: usize) -> i16 {
        if score > VALUE_MATE_THRESHOLD {
            score + ply as i16
        } else if score < -VALUE_MATE_THRESHOLD {
            score - ply as i16
        } else {
            score
        }
    }

    #[inline(always)]
    fn score_from_tt(score: i16, ply: usize) -> i16 {
        if score > VALUE_MATE_THRESHOLD {
            score - ply as i16
        } else if score < -VALUE_MATE_THRESHOLD {
            score + ply as i16
        } else {
            score
        }
    }

    /// Tra cứu TT: trả về (best_move, tt_score_cutoff_if_usable)
    #[inline(always)]
    pub fn probe(
        &self,
        key: u64,
        ply: usize,
        depth: i8,
        alpha: i16,
        beta: i16,
    ) -> (Option<Move>, Option<i16>) {
        let idx = (key as usize) & self.mask;
        let entry = &self.entries[idx];

        if entry.key != key {
            return (None, None);
        }

        let best_move = if entry.best_move != 0 {
            Some(Move(entry.best_move))
        } else {
            None
        };

        // Nếu độ sâu trong TT đủ lớn, kiểm tra xem có thể cutoff trực tiếp hay không
        if entry.depth >= depth {
            let score = Self::score_from_tt(entry.score, ply);

            match entry.flag {
                TT_FLAG_EXACT => return (best_move, Some(score)),
                TT_FLAG_LOWER => {
                    if score >= beta {
                        return (best_move, Some(score));
                    }
                }
                TT_FLAG_UPPER => {
                    if score <= alpha {
                        return (best_move, Some(score));
                    }
                }
                _ => {}
            }
        }

        (best_move, None)
    }

    /// Lưu kết quả tìm kiếm vào TT
    #[inline(always)]
    pub fn record(
        &mut self,
        key: u64,
        ply: usize,
        depth: i8,
        score: i16,
        flag: u8,
        best_move: Option<Move>,
    ) {
        let idx = (key as usize) & self.mask;
        let current_entry = &self.entries[idx];

        // Chính sách thay thế: ghi đè nếu khác khóa, hoặc độ sâu mới >= độ sâu cũ, hoặc thế hệ cũ
        let should_replace = current_entry.key != key
            || depth >= current_entry.depth
            || self.age != current_entry.age;

        if should_replace {
            let move_val = match best_move {
                Some(m) => m.0,
                None => {
                    if current_entry.key == key {
                        current_entry.best_move
                    } else {
                        0
                    }
                }
            };

            self.entries[idx] = TTEntry {
                key,
                best_move: move_val,
                score: Self::score_to_tt(score, ply),
                depth,
                flag,
                age: self.age,
            };
        }
    }
}
