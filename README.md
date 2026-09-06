# Wisdom NNUE 象棋 🚀⚡️

**Wisdom NNUE** là một cờ máy (Xiangqi Chess Engine) hiệu năng cao viết bằng **100% Pure Rust**, kết hợp mạng nơ-ron đánh giá vị trí **NNUE (Efficiently Updatable Neural Network)** theo kiến trúc HalfKP chuẩn Stockfish và bộ tìm kiếm **Alpha-Beta (PVS / Negamax)** tối ưu hóa phần cứng native CPU (AVX-512 / AVX2).

---

## 📑 Mục Lục
1. [Tính Năng Nổi Bật](#1-tính-năng-nổi-bật)
2. [Cài Đặt & Biên Dịch](#2-cài-đặt--biên-dịch)
3. [Hướng Dẫn Training Pipeline (Quy Trình Tự Huấn Luyện)](#3-hướng-dẫn-training-pipeline)
   - [Tổng quan quy trình khép kín](#tổng-quan-quy-trình-khép-kín)
   - [Bước 1: Khởi tạo mô hình (Scratch)](#bước-1-khởi-tạo-mô-hình-scratch)
   - [Bước 2: Sinh dữ liệu tự chơi (Self-Play .bin)](#bước-2-sinh-dữ-liệu-tự-chơi-self-play-bin)
   - [Bước 3: Huấn luyện liên tục (Continuous Training)](#bước-3-huấn-luyện-liên-tục-continuous-training)
   - [Bước 4: Đấu thẩm định sức cờ (Arena)](#bước-4-đấu-thẩm-định-sức-cờ-arena)
   - [Bước 5: Trực quan hóa & Thi đấu trên GUI](#bước-5-trực-quan-hóa--thi-đấu-trên-gui)
4. [Cấu Trúc Dữ Liệu Nhị Phân & FIFO Replay Buffer](#4-cấu-trúc-dữ-liệu-nhị-phân--fifo-replay-buffer)
5. [Giao Thức UCCI / Kết Nối GUI Bên Ngoài](#5-giao-thức-ucci--kết-nối-gui-bên-ngoài)

---

## 1. Tính Năng Nổi Bật 🌟

* **Tốc độ tìm kiếm vượt trội**: Đạt **~460,000 NPS** trên 1 nhân CPU nhờ tối ưu hóa lệnh máy Native SIMD (AVX-512 VNNI, AVX2) và Fat LTO.
* **Bộ tìm kiếm Alpha-Beta PVS chuẩn Stockfish**:
  * Tối ưu hóa chuyển dịch Accumulator vi phân $O(1)$.
  * Transposition Table căn chỉnh 16-byte (16-byte aligned) khớp CPU Cache Line.
  * Late Move Pruning (LMP), Futility Pruning (FP), Reverse Futility Pruning (RFP).
  * Logarithmic Late Move Reductions (LMR), Quiescence Search (QS), Aspiration Windows.
  * Phán xử luật lặp cấm chuẩn Cờ Tướng quốc tế (Perpetual Check, Perpetual Chase).
* **Định dạng nhị phân siêu nén (`.bin`)**: Mỗi vị trí cờ được nén chính xác **96 bytes** (bội số 16-byte), nạp vào PyTorch qua `np.memmap` tốn đúng **0.00 giây**.
* **Replay Buffer First-In First-Out (FIFO)**: Tự động đào thải dữ liệu cũ nhất khi vượt ngưỡng chuẩn (mặc định 1,000,000 thế cờ ~ 96 MB), bảo đảm kích thước đĩa cố định và model luôn học từ dữ liệu chất lượng nhất.
* **Top-N Move Selection toàn ván**: Lựa chọn ngẫu nhiên trong Top 3 nước cờ có điểm số cao nhất xuyên suốt ván đấu, tạo ra không gian phân nhánh khổng lồ mà không bao giờ đi nước ngáo.
* **Giao diện đồ họa Macroquad tích hợp**: Hỗ trợ hiển thị thanh ưu thế (Advantage Bar), điểm centipawns, chuỗi nước đi tối ưu (PV Line) và điều chỉnh độ sâu 4-24 plies mượt mà 60 FPS.

---

## 2. Cài Đặt & Biên Dịch 🛠️

### Yêu cầu hệ thống:
* **Rust**: Phiên bản 1.80+ (khuyên dùng bản stable mới nhất).
* **Python**: 3.10+ cùng các thư viện: `torch`, `numpy`, `pandas`.

### Biên dịch tất cả các binary:
```bash
# Biên dịch ở chế độ Release với tối ưu hóa Native CPU:
cargo build --release --bins
```

Các binary sinh ra trong thư mục `target/release/`:
* `wisdom`: Engine chính hỗ trợ chuẩn giao thức UCCI/UCI.
* `gui`: Giao diện đồ họa bàn cờ tương tác trực tiếp với engine NNUE.
* `selfplay`: Công cụ sinh dữ liệu tự chơi đa luồng ra file nhị phân `.bin`.
* `arena`: Sàn đấu kiểm tra Elo / tỉ lệ thắng giữa 2 phiên bản hoặc 2 độ sâu.

---

## 3. Hướng Dẫn Training Pipeline 🧠♟️

### Tổng quan quy trình khép kín:
```text
  ┌────────────────────────────────────────────────────────┐
  │                 Khởi tạo model (Scratch)               │
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │  Self-Play Generator (Rust, 8 luồng, Top-3 nước tốt)   │
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼  (Ghi tiếp tục & cắt tỉa FIFO)
  ┌────────────────────────────────────────────────────────┐
  │      wisdom_data/replay_buffer.bin (Chuẩn 1M FENs)     │
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼  (Nạp qua memmap trong 0.00s)
  ┌────────────────────────────────────────────────────────┐
  │  train_nnue.py (Fine-tuning / AVX-512 / Iris Xe GPU)   │
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼  (Tự động lượng hóa int8/int16)
  ┌────────────────────────────────────────────────────────┐
  │         wisdom_models/wisdom_nnue.bin (5.55 MB)        │
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │       Thẩm định sức cờ trên Arena & Trải nghiệm GUI    │
  └────────────────────────────────────────────────────────┘
```

---

### Bước 1: Khởi tạo mô hình (Scratch)
Nếu bạn chưa có model hoặc muốn làm mới hoàn toàn từ đầu:
```bash
python3 train_nnue.py --init-only
```
Lệnh này sẽ:
1. Khởi tạo trọng số ngẫu nhiên chuẩn NNUE cho mô hình `WisdomNNUE`.
2. Lưu checkpoint PyTorch vào: `wisdom_models/wisdom_nnue_best.pt`.
3. Tự động lượng hóa và xuất file nhị phân: `wisdom_models/wisdom_nnue.bin`.

---

### Bước 2: Sinh dữ liệu tự chơi (Self-Play .bin)
Sử dụng binary `selfplay` đa luồng để tự sinh các ván đấu chất lượng cao:

```bash
cargo run --release --bin selfplay -- \
  --games 1000 \
  --depth 7 \
  --threads 8 \
  --random-plies 3 \
  --buffer-size 1000000 \
  --output wisdom_data/replay_buffer.bin
```

#### Giải thích các tham số:
* `--games 1000`: Số ván đấu cần sinh trong đợt này (thường từ 500 đến 1,000 ván).
* `--depth 7`: Độ sâu tìm kiếm Alpha-Beta (depth 7 mất ~20ms mỗi nước, bảo đảm tính chiến thuật sắc bén).
* `--threads 8`: Số luồng CPU chạy song song (tận dụng tối đa đa nhân của máy).
* `--random-plies 3` (hoặc `--random-top 3`): **Ở mọi nước đi trong ván**, engine tính toán điểm số và chọn ngẫu nhiên 1 trong **Top 3 nước có điểm số cao nhất**. Nước cờ luôn chuẩn xác mà không bị trùng lặp biến thế.
* `--buffer-size 1000000`: Dung lượng tối đa của Replay Buffer FIFO (chuẩn 1 triệu thế cờ ~ 96 MB). Khi vượt quá, engine tự động cắt bỏ các thế cờ cũ nhất ở đầu file.
* `--output`: Đường dẫn lưu file nhị phân.

---

### Bước 3: Huấn luyện liên tục (Continuous Training)
Sau khi sinh dữ liệu xong, chạy script Python để hấp thụ dữ liệu mới:

```bash
python3 train_nnue.py --data wisdom_data/replay_buffer.bin --epochs 5
```

#### Cơ chế tự động thông minh:
* **Tự động Resume / Fine-Tuning**: Script tự động phát hiện `wisdom_models/wisdom_nnue_best.pt`, nạp trọng số cũ và hạ Learning Rate xuống mức tinh chỉnh ($3 \times 10^{-4}$) để không làm phá vỡ kiến thức đã học.
* **Tự động nhận diện phần cứng**:
  * Ưu tiên GPU Intel Iris Xe (`torch.xpu` qua Level-Zero driver).
  * Tối ưu hóa CPU đa luồng Intel Tiger Lake với tập lệnh **AVX-512 VNNI + MKL-DNN / oneDNN** (tốc độ train dưới **0.5s / epoch**).
* **Tự động lượng hóa**: Khi train xong, script tự động lượng hóa và ghi đè file `wisdom_models/wisdom_nnue.bin`. Engine và GUI sẽ lập tức nhận diện trí thông minh mới mà không cần thao tác thủ công!

*(Mẹo: Nếu bạn muốn ép buộc train từ đầu mà không kế thừa model cũ, hãy thêm cờ `--init-scratch`)*:
```bash
python3 train_nnue.py --data wisdom_data/replay_buffer.bin --epochs 10 --init-scratch
```

---

### Bước 4: Đấu thẩm định sức cờ (Arena)
Cho model mới thi đấu để đánh giá độ ổn định và tỉ lệ thắng:

```bash
# Đấu 50 ván, Đỏ depth 7 vs Đen depth 7:
cargo run --release --bin arena 50 7 7
```
Output sẽ thống kê chi tiết: số ván Đỏ thắng, Đen thắng, Hòa cờ, và số ván vi phạm luật lặp cấm (Perpetual rules).

---

### Bước 5: Trực quan hóa & Thi đấu trên GUI
Khởi động bàn cờ giao diện đồ họa để chiến thử với AI:

```bash
cargo run --release --bin gui
```
* **Advantage Bar**: Hiển thị thanh ưu thế thời gian thực (Xanh: người chơi ưu thế, Đỏ: engine ưu thế).
* **Centipawns & PV**: Xem trực tiếp điểm đánh giá và chuỗi nước đi AI dự đoán.
* **Depth Control**: Nút `+` / `-` điều chỉnh độ sâu tìm kiếm từ 4 đến 24 plies linh hoạt.

---

## 4. Cấu Trúc Dữ Liệu Nhị Phân & FIFO Replay Buffer 📦

### Cấu trúc Record 96 Bytes (`WisdomBinRecord`):
```rust
#[repr(C, packed)]
pub struct WisdomBinRecord {
    pub squares: [u8; 90], // 10 hàng x 9 cột (0..6: Đỏ, 7..13: Đen, 255: Trống)
    pub side_to_move: u8,  // 0: Đỏ, 1: Đen
    pub score: i16,        // Centipawns (-30000..30000)
    pub outcome: i8,       // 1: Thắng, 0: Hòa, -1: Thua (theo góc nhìn STM)
    pub ply: u16,          // Nước đi thứ mấy
} // Kích thước: Đúng 96 bytes (bội số 16-byte aligned)
```

### Công thức Mục Tiêu Huấn Luyện ($\lambda$-Blending):
$$\text{Target} = 0.8 \times \tanh\left(\frac{\text{SearchScore}}{600}\right) + 0.2 \times \text{GameOutcome}$$
* $80\%$ từ Search Score: Cung cấp độ chi tiết vi mô cho từng nước đi (hơn quân, chiếm lộ đẹp).
* $20\%$ từ Game Outcome: Mỏ neo thực tế sửa sai nếu nước cờ dẫn đến thua cuộc.
* Khi tìm ra chiếu bí sát cục (Mate 100%), điểm chạm trần tuyệt đối $\pm 1.0000$.

### Cơ chế FIFO (First-In First-Out):
* Kích thước chuẩn: **1,000,000 positions** ($\approx 96\text{ MB}$, tương đương $12,000 - 15,000$ ván cờ).
* Dữ liệu mới luôn được ghi vào cuối buffer.
* Khi buffer vượt ngưỡng 1 triệu thế cờ, hệ thống tự động cắt bỏ phần dữ liệu cũ nhất ở đầu file. Mô hình không bao giờ bị dính dữ liệu lỗi thời của các đời engine yếu!

---

## 5. Giao Thức UCCI / Kết Nối GUI Bên Ngoài ♟️🎮

Engine `wisdom` hỗ trợ đầy đủ giao thức chuẩn **UCCI** (Universal Chinese Chess Protocol) và **UCI**, cho phép kết nối trực tiếp vào các phần mềm GUI Cờ Tướng chuyên nghiệp (BHGUI, Shark, Saola, CCBridge):

```bash
# Khởi chạy engine ở chế độ dòng lệnh UCCI:
cargo run --release --bin wisdom
```

### Các lệnh UCCI cơ bản:
```text
ucci
isready
position startpos moves h2e2 h9g7
go depth 12
quit
```

Để thêm vào các GUI như BHGUI/Shark: Bạn chỉ cần trỏ đường dẫn engine đến file:
`/home/huy/Documents/wisdom/target/release/wisdom`

---

Chúc bạn có những giờ phút nghiên cứu và thi đấu cờ tướng thật thú vị với **Wisdom NNUE**! 🚀🏆
