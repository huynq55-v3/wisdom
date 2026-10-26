import os
import sys
import time
import struct
import shutil
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# =====================================================================
# 1. HẰNG SỐ ĐẶC TRƯNG NNUE CỜ TƯỚNG (HalfKP)
# =====================================================================
# Bàn cờ: 10 hàng x 9 cột = 90 ô
# Cung Tướng phe ta: 3 hàng x 3 cột = 9 ô (hàng 7..9, cột 3..5)
# Quân cờ: 14 loại (0..6 quân ta: K, A, B, N, R, C, P; 7..13 quân địch: k, a, b, n, r, c, p)
NUM_KING_SQ = 9
NUM_PIECE_TYPES = 14
NUM_BOARD_SQ = 90
NUM_FEATURES = NUM_KING_SQ * NUM_PIECE_TYPES * NUM_BOARD_SQ  # 9 * 14 * 90 = 11,340
PADDING_IDX = NUM_FEATURES
MAX_PIECES_PER_BOARD = 32

L1_DIM = 512
L2_DIM = 32
L3_DIM = 32

PIECE_MAP = {
    'K': 0, 'A': 1, 'B': 2, 'N': 3, 'R': 4, 'C': 5, 'P': 6,
    'k': 7, 'a': 8, 'b': 9, 'n': 10, 'r': 11, 'c': 12, 'p': 13
}

# =====================================================================
# 2. CẤU TRÚC RECORD NHỊ PHÂN (96 BYTES) CHO SELF-PLAY
# =====================================================================
RECORD_DTYPE = np.dtype([
    ('squares', np.uint8, 90),
    ('side_to_move', np.uint8),
    ('score', np.int16),
    ('outcome', np.int8),
    ('ply', np.uint16)
])

# =====================================================================
# 3. TRÍCH XUẤT ĐẶC TRƯNG (TỪ MẢNG 90 Ô & TỪ CHUỖI FEN)
# =====================================================================
def squares_to_features(squares, stm, is_mirror=False):
    """
    Trích xuất đặc trưng HalfKP trực tiếp từ mảng 90 ô cờ (siêu tốc, zero overhead).
    """
    red_k = None
    black_k = None
    pieces = []

    for r in range(10):
        for c in range(9):
            p = int(squares[r * 9 + c])
            if p != 255:
                actual_c = 8 - c if is_mirror else c
                pieces.append((r, actual_c, p))
                if p == 0:
                    red_k = (r, actual_c)
                elif p == 7:
                    black_k = (r, actual_c)

    if red_k is None:
        red_k = (9, 4)
    if black_k is None:
        black_k = (0, 4)

    # 1. Tướng Đỏ trong Cung Đỏ (hàng 7..9, cột 3..5)
    kr_row = max(7, min(9, red_k[0]))
    kr_col = max(3, min(5, red_k[1]))
    kr_idx = (kr_row - 7) * 3 + (kr_col - 3)

    # 2. Tướng Đen theo góc nhìn Đen (xoay 180 độ)
    kb_rot_r = 9 - black_k[0]
    kb_rot_c = 8 - black_k[1]
    kb_rot_r = max(7, min(9, kb_rot_r))
    kb_rot_c = max(3, min(5, kb_rot_c))
    kb_idx = (kb_rot_r - 7) * 3 + (kb_rot_c - 3)

    feat_red = []
    feat_black = []

    for r, c, p in pieces:
        # Góc nhìn Đỏ
        sq_red = r * 9 + c
        feat_red.append(kr_idx * 1260 + p * 90 + sq_red)

        # Góc nhìn Đen
        r_rot = 9 - r
        c_rot = 8 - c
        sq_black = r_rot * 9 + c_rot
        p_black = p - 7 if p >= 7 else p + 7
        feat_black.append(kb_idx * 1260 + p_black * 90 + sq_black)

    while len(feat_red) < MAX_PIECES_PER_BOARD:
        feat_red.append(PADDING_IDX)
    while len(feat_black) < MAX_PIECES_PER_BOARD:
        feat_black.append(PADDING_IDX)

    if len(feat_red) > MAX_PIECES_PER_BOARD:
        feat_red = feat_red[:MAX_PIECES_PER_BOARD]
    if len(feat_black) > MAX_PIECES_PER_BOARD:
        feat_black = feat_black[:MAX_PIECES_PER_BOARD]

    # Bên tới lượt đi (STM) luôn là US, đối thủ là THEM
    if stm == 0:
        return feat_red, feat_black
    else:
        return feat_black, feat_red


def fen_to_features(fen_str, is_mirror=False):
    """
    Chuyển FEN sang 2 danh sách feature indices (Us, Them) cho NNUE.
    """
    parts = fen_str.split()
    board_part = parts[0]
    stm = parts[1].lower() if len(parts) > 1 else 'w'

    grid = []
    for row_str in board_part.split('/'):
        row = []
        for ch in row_str:
            if ch.isdigit():
                row.extend([-1] * int(ch))
            else:
                row.append(PIECE_MAP[ch])
        grid.append(row)

    red_k = None
    black_k = None
    pieces = []

    for r in range(10):
        for c in range(9):
            p = grid[r][c]
            if p != -1:
                actual_c = 8 - c if is_mirror else c
                pieces.append((r, actual_c, p))
                if p == 0:
                    red_k = (r, actual_c)
                elif p == 7:
                    black_k = (r, actual_c)

    if red_k is None:
        red_k = (9, 4)
    if black_k is None:
        black_k = (0, 4)

    kr_row = max(7, min(9, red_k[0]))
    kr_col = max(3, min(5, red_k[1]))
    kr_idx = (kr_row - 7) * 3 + (kr_col - 3)

    kb_rot_r = 9 - black_k[0]
    kb_rot_c = 8 - black_k[1]
    kb_rot_r = max(7, min(9, kb_rot_r))
    kb_rot_c = max(3, min(5, kb_rot_c))
    kb_idx = (kb_rot_r - 7) * 3 + (kb_rot_c - 3)

    feat_red = []
    feat_black = []

    for r, c, p in pieces:
        sq_red = r * 9 + c
        feat_red.append(kr_idx * 1260 + p * 90 + sq_red)

        r_rot = 9 - r
        c_rot = 8 - c
        sq_black = r_rot * 9 + c_rot
        p_black = p - 7 if p >= 7 else p + 7
        feat_black.append(kb_idx * 1260 + p_black * 90 + sq_black)

    while len(feat_red) < MAX_PIECES_PER_BOARD:
        feat_red.append(PADDING_IDX)
    while len(feat_black) < MAX_PIECES_PER_BOARD:
        feat_black.append(PADDING_IDX)

    if len(feat_red) > MAX_PIECES_PER_BOARD:
        feat_red = feat_red[:MAX_PIECES_PER_BOARD]
    if len(feat_black) > MAX_PIECES_PER_BOARD:
        feat_black = feat_black[:MAX_PIECES_PER_BOARD]

    if stm == 'w':
        return feat_red, feat_black
    else:
        return feat_black, feat_red


# =====================================================================
# 4. DATASETS: HỖ TRỢ CẢ FILE .BIN (SIÊU TỐC) VÀ FILE .CSV
# =====================================================================
class WisdomBinDataset(Dataset):
    def __init__(self, bin_file, max_samples=None):
        print(f"📦 Đang nạp dataset nhị phân từ: {bin_file}...")
        t0 = time.time()
        file_size = os.path.getsize(bin_file)
        num_records = file_size // 96
        if num_records == 0:
            raise ValueError(f"File nhị phân rỗng hoặc kích thước không hợp lệ: {bin_file}")

        self.data = np.memmap(bin_file, dtype=RECORD_DTYPE, mode='r', shape=(num_records,))
        if max_samples and max_samples < num_records:
            indices = np.random.choice(num_records, max_samples, replace=False)
            self.data = self.data[indices]

        self.len = len(self.data)
        print(f"✅ Đã nạp thành công {self.len:,} positions nhị phân trong {time.time() - t0:.2f}s!")

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        rec = self.data[idx]
        squares = rec['squares']
        stm = int(rec['side_to_move'])
        score = float(rec['score'])
        outcome = float(rec['outcome'])

        # Chuẩn hóa search score về [-1.0, 1.0] tương ứng với không gian đầu ra tanh của WisdomNNUE.
        # Ở Rust engine: evaluate_cp = evaluate() * 1000.0 nên score_val = score / 1000.0.
        # Tuyệt đối không dùng thêm hàm np.tanh(score / 600.0) để tránh lỗi Double Tanh Compression.
        if score >= 10000:
            score_val = 1.0
        elif score <= -10000:
            score_val = -1.0
        else:
            score_val = np.clip(score / 1000.0, -1.0, 1.0)

        # Target blend: kết hợp giữa search evaluation (80%) và kết quả game (20%)
        target = 0.8 * score_val + 0.2 * outcome
        target = np.float32(np.clip(target, -1.0, 1.0))

        # Data Augmentation: 50% cơ hội lật gương ngang trái-phải
        is_mirror = (np.random.rand() < 0.5)
        us_feats, them_feats = squares_to_features(squares, stm, is_mirror=is_mirror)

        return (
            torch.tensor(us_feats, dtype=torch.long),
            torch.tensor(them_feats, dtype=torch.long),
            torch.tensor(target, dtype=torch.float32)
        )


class XiangqiNNUEDataset(Dataset):
    def __init__(self, csv_file, max_samples=None):
        print(f"📦 Đang nạp dataset CSV từ: {csv_file}...")
        t0 = time.time()
        df = pd.read_csv(
            csv_file,
            header=None,
            usecols=[0, 1],
            names=['fen', 'value'],
            dtype={'fen': str, 'value': np.float32}
        )
        valid_mask = df['fen'].notna() & df['value'].notna()
        df = df[valid_mask].reset_index(drop=True)

        if max_samples and max_samples < len(df):
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

        self.fens = df['fen'].values
        self.values = df['value'].values
        print(f"✅ Đã nạp thành công {len(self.fens):,} positions CSV trong {time.time() - t0:.2f}s!")

    def __len__(self):
        return len(self.fens)

    def __getitem__(self, idx):
        fen = self.fens[idx]
        val = np.float32(np.clip(self.values[idx], -1.0, 1.0))
        is_mirror = (np.random.rand() < 0.5)
        us_feats, them_feats = fen_to_features(fen, is_mirror=is_mirror)

        return (
            torch.tensor(us_feats, dtype=torch.long),
            torch.tensor(them_feats, dtype=torch.long),
            torch.tensor(val, dtype=torch.float32)
        )


# =====================================================================
# 5. MÔ HÌNH WISDOM NNUE
# =====================================================================
class ClippedReLU(nn.Module):
    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0)


class WisdomNNUE(nn.Module):
    def __init__(self, l1=L1_DIM, l2=L2_DIM, l3=L3_DIM):
        super().__init__()
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3

        self.ft = nn.EmbeddingBag(NUM_FEATURES + 1, l1, mode='sum', padding_idx=PADDING_IDX)
        self.ft_bias = nn.Parameter(torch.zeros(l1))
        self.act = ClippedReLU()

        self.fc0 = nn.Linear(l1 * 2, l2)
        self.fc1 = nn.Linear(l2, l3)
        self.fc2 = nn.Linear(l3, 1)

    def forward(self, us_indices, them_indices):
        acc_us = self.act(self.ft(us_indices) + self.ft_bias)
        acc_them = self.act(self.ft(them_indices) + self.ft_bias)
        x = torch.cat([acc_us, acc_them], dim=1)
        x = self.act(self.fc0(x))
        x = self.act(self.fc1(x))
        out = torch.tanh(self.fc2(x))
        return out


# =====================================================================
# 6. XUẤT MÔ HÌNH LƯỢNG TỬ HÓA (.BIN)
# =====================================================================
def export_quantized_binary(model, output_path="wisdom_models/wisdom_nnue.bin"):
    print(f"\n📦 Đang xuất mô hình lượng tử hóa ra: {output_path}...")
    model.eval().cpu()

    QA = 255.0
    QB = 64.0

    ft_weight = model.ft.weight.data[:NUM_FEATURES, :].numpy()
    ft_bias = model.ft_bias.data.numpy()

    ft_weight_i16 = np.round(np.clip(ft_weight * QA, -32767, 32767)).astype(np.int16)
    ft_bias_i16 = np.round(np.clip(ft_bias * QA, -32767, 32767)).astype(np.int16)

    fc0_weight = model.fc0.weight.data.numpy()
    fc0_bias = model.fc0.bias.data.numpy()
    fc0_weight_i8 = np.round(np.clip(fc0_weight * QB, -127, 127)).astype(np.int8)
    fc0_bias_i32 = np.round(fc0_bias * (QA * QB)).astype(np.int32)

    fc1_weight = model.fc1.weight.data.numpy()
    fc1_bias = model.fc1.bias.data.numpy()
    fc1_weight_i8 = np.round(np.clip(fc1_weight * QB, -127, 127)).astype(np.int8)
    fc1_bias_i32 = np.round(fc1_bias * (QB * QB)).astype(np.int32)

    fc2_weight_f32 = model.fc2.weight.data.numpy().astype(np.float32)
    fc2_bias_f32 = model.fc2.bias.data.numpy().astype(np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as f:
        magic = b"WSDMNNUE"
        version = 1
        header = struct.pack("<8sIIIII", magic, version, L1_DIM, L2_DIM, L3_DIM, NUM_FEATURES)
        f.write(header)

        f.write(ft_bias_i16.tobytes())
        f.write(ft_weight_i16.tobytes())

        f.write(fc0_bias_i32.tobytes())
        f.write(fc0_weight_i8.tobytes())

        f.write(fc1_bias_i32.tobytes())
        f.write(fc1_weight_i8.tobytes())

        f.write(fc2_bias_f32.tobytes())
        f.write(fc2_weight_f32.tobytes())

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Hoàn tất xuất file: {output_path} (Kích thước: {file_size_mb:.2f} MB)")


def init_model_scratch(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model = WisdomNNUE()
    # Khởi tạo trọng số chuẩn NNUE: std=0.08 để khi lượng tử hóa QA=255.0 không bị triệt tiêu neuron
    nn.init.normal_(model.ft.weight, mean=0.0, std=0.08)
    nn.init.zeros_(model.ft_bias)
    nn.init.kaiming_normal_(model.fc0.weight, nonlinearity='relu')
    nn.init.zeros_(model.fc0.bias)
    nn.init.kaiming_normal_(model.fc1.weight, nonlinearity='relu')
    nn.init.zeros_(model.fc1.bias)
    nn.init.xavier_normal_(model.fc2.weight)
    nn.init.zeros_(model.fc2.bias)

    pt_path = os.path.join(output_dir, "wisdom_nnue_best.pt")
    torch.save(model.state_dict(), pt_path)
    print(f"✨ Đã khởi tạo mô hình mới từ Scratch: {pt_path}")

    bin_path = os.path.join(output_dir, "wisdom_nnue.bin")
    export_quantized_binary(model, bin_path)
    return model


# =====================================================================
# 7. PHÁT HIỆN PHẦN CỨNG (INTEL IRIS XE / AVX-512 CPU)
# =====================================================================
def select_compute_device():
    # 1. Thử Intel XPU (Iris Xe)
    try:
        import intel_extension_for_pytorch as ipex
    except ImportError:
        pass

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        dev_name = torch.xpu.get_device_name(0) if hasattr(torch.xpu, "get_device_name") else "Intel GPU"
        print(f"🚀 Thiết bị tăng tốc: Intel Iris Xe GPU (XPU: {dev_name}) ⚡️")
        return torch.device("xpu")

    # 2. Thử NVIDIA CUDA (nếu có)
    if torch.cuda.is_available():
        print(f"🚀 Thiết bị tăng tốc: NVIDIA GPU ({torch.cuda.get_device_name(0)}) ⚡️")
        return torch.device("cuda")

    # 3. Mặc định CPU Intel AVX-512 + MKL / oneDNN
    num_threads = os.cpu_count() or 4
    torch.set_num_threads(num_threads)
    print(f"🚀 Thiết bị: CPU Intel Tiger Lake (Tối ưu hóa AVX-512 VNNI / oneDNN, {num_threads} threads) ⚡️")
    return torch.device("cpu")


# =====================================================================
# 8. VÒNG LẶP HUẤN LUYỆN CHÍNH (MAIN PIPELINE)
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Wisdom NNUE Continuous Training Pipeline")
    parser.add_argument("--data", type=str, default="wisdom_data/replay_buffer.bin", help="Đường dẫn file .bin hoặc .csv")
    parser.add_argument("--epochs", type=int, default=10, help="Số lượng epoch")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (mặc định: 1e-3 nếu scratch, 3e-4 nếu resume)")
    parser.add_argument("--output-dir", type=str, default="wisdom_models", help="Thư mục lưu model")
    parser.add_argument("--resume", type=str, default=None, help="Đường dẫn file checkpoint .pt cụ thể để tiếp tục huấn luyện (ví dụ: wisdom_models/archive/gen1.pt)")
    parser.add_argument("--name", type=str, default="wisdom_nnue", help="Tên model khi lưu kết quả (mặc định: wisdom_nnue)")
    parser.add_argument("--init-scratch", action="store_true", help="Bắt buộc khởi tạo model mới từ đầu thay vì resume")
    parser.add_argument("--init-only", action="store_true", help="Chỉ khởi tạo model scratch rồi thoát, không train")
    parser.add_argument("--max-samples", type=int, default=None, help="Giới hạn số mẫu nạp để test nhanh")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model_name = args.name
    pt_path = os.path.join(args.output_dir, f"{model_name}_best.pt")
    bin_path = os.path.join(args.output_dir, f"{model_name}.bin")

    # Nếu chỉ yêu cầu tạo model khởi điểm
    if args.init_only:
        print(f"🌱 Đang tạo mô hình khởi điểm từ Scratch ({model_name})...")
        init_model_scratch(args.output_dir)
        print("✅ Hoàn tất khởi tạo mô hình!")
        return

    device = select_compute_device()

    # 1. Quản lý mô hình: Scratch vs Continuous Training (Resume)
    is_resumed = False
    default_resume_pt = os.path.join(args.output_dir, "wisdom_nnue_best.pt")

    if args.init_scratch:
        print("🌱 Bắt đầu huấn luyện mô hình MỚI HOÀN TOÀN (From Scratch)...")
        model = init_model_scratch(args.output_dir).to(device)
        default_lr = 0.001
    elif args.resume is not None:
        if not os.path.exists(args.resume):
            print(f"❌ Không tìm thấy file checkpoint chỉ định: {args.resume}")
            sys.exit(1)
        print(f"🔄 Tiếp tục huấn luyện từ checkpoint chỉ định: {args.resume}")
        model = WisdomNNUE().to(device)
        state_dict = torch.load(args.resume, map_location=device, weights_only=True)
        try:
            model.load_state_dict(state_dict)
            default_lr = 0.0003  # Learning rate nhỏ hơn để fine-tune
            is_resumed = True
        except RuntimeError as e:
            print(f"⚠️ Checkpoint chỉ định {args.resume} có cấu trúc kích thước khác với kiến trúc mới (L1={L1_DIM})!")
            print(f"🌱 Tự động khởi tạo mô hình mới (From Scratch) với L1={L1_DIM}...")
            model = init_model_scratch(args.output_dir).to(device)
            default_lr = 0.001
            is_resumed = False
    elif os.path.exists(default_resume_pt):
        print(f"🔄 Phát hiện checkpoint mặc định tại: {default_resume_pt}")
        model = WisdomNNUE().to(device)
        state_dict = torch.load(default_resume_pt, map_location=device, weights_only=True)
        try:
            model.load_state_dict(state_dict)
            print("🚀 Tiếp tục huấn luyện (Fine-Tuning / Warm-Start) từ model cũ!")
            default_lr = 0.0003  # Learning rate nhỏ hơn để fine-tune
            is_resumed = True
        except RuntimeError as e:
            print(f"⚠️ Checkpoint cũ {default_resume_pt} là kiến trúc cũ (L1=256), không khớp với kiến trúc mới (L1={L1_DIM})!")
            print(f"🌱 Tự động khởi tạo mô hình mới (From Scratch) với L1={L1_DIM}...")
            model = init_model_scratch(args.output_dir).to(device)
            default_lr = 0.001
            is_resumed = False
    else:
        print("⚠️ Không tìm thấy checkpoint cũ. Khởi tạo mô hình mới từ Scratch...")
        model = init_model_scratch(args.output_dir).to(device)
        default_lr = 0.001

    learning_rate = args.lr if args.lr is not None else default_lr
    print(f"🎯 Learning Rate: {learning_rate:.6f}")

    # 2. Nạp dữ liệu
    if not os.path.exists(args.data):
        print(f"❌ Không tìm thấy file dữ liệu: {args.data}")
        print("💡 Hãy chạy selfplay generator trước để sinh dữ liệu: cargo run --release --bin selfplay")
        sys.exit(1)

    if args.data.endswith(".bin"):
        dataset = WisdomBinDataset(args.data, max_samples=args.max_samples)
    else:
        dataset = XiangqiNNUEDataset(args.data, max_samples=args.max_samples)

    val_size = max(1, int(len(dataset) * 0.05))
    if val_size >= len(dataset):
        val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    num_workers = 4 if os.name != 'nt' and device.type != 'xpu' else 0
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == 'cuda'))

    print(f"📊 Tập Train: {train_size:,} samples | Tập Val: {val_size:,} samples")

    criterion = nn.HuberLoss(delta=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_loss = float('inf')

    # 3. Vòng lặp huấn luyện
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        start_time = time.time()

        for batch_idx, (us, them, targets) in enumerate(train_loader):
            us = us.to(device, non_blocking=True)
            them = them.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad()
            preds = model(us, them)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * us.size(0)

        scheduler.step()
        train_loss /= train_size
        elapsed = time.time() - start_time

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for us, them, targets in val_loader:
                us = us.to(device, non_blocking=True)
                them = them.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True).unsqueeze(1)

                preds = model(us, them)
                val_loss += criterion(preds, targets).item() * us.size(0)

        val_loss /= val_size
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch [{epoch+1:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | LR: {current_lr:.6f} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

        # Lưu checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), pt_path)
            print(f"   ⭐️ Đã lưu checkpoint tốt nhất vào: {pt_path}")

    # 4. Xuất file binary lượng tử hóa cho Rust engine
    export_quantized_binary(model, bin_path)
    if model_name != "wisdom_nnue":
        default_bin = os.path.join(args.output_dir, "wisdom_nnue.bin")
        default_pt = os.path.join(args.output_dir, "wisdom_nnue_best.pt")
        shutil.copyfile(bin_path, default_bin)
        if os.path.exists(pt_path):
            shutil.copyfile(pt_path, default_pt)
        print(f"🔗 Đồng thời cập nhật vào model mặc định: {default_bin}")
    print("\n🎉 HUẤN LUYỆN HOÀN TẤT THÀNH CÔNG! ĐÃ CẬP NHẬT MÔ HÌNH VÀO WISDOM ENGINE! 🚀")


if __name__ == "__main__":
    main()
