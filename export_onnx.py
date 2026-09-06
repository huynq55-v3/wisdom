import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

# =====================================================================
# 1. KIẾN TRÚC MẠNG RESNET (8 BLOCKS, 8100 POLICY)
# =====================================================================
class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class XiangqiNet(nn.Module):
    def __init__(self, num_res_blocks=8, channels=128):
        super(XiangqiNet, self).__init__()
        self.conv_input = nn.Conv2d(14, channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(channels)
        self.res_blocks = nn.Sequential(*[ResBlock(channels) for _ in range(num_res_blocks)])

        self.conv_policy = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_head = nn.Linear(2 * 10 * 9, 8100)

        self.fc1 = nn.Linear(channels, 64)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.bn_input(self.conv_input(x)))
        x_spatial = self.res_blocks(x)

        x_pol = self.conv_policy(x_spatial).view(batch_size, -1)
        logits_policy = self.policy_head(x_pol)

        x_val = x_spatial.view(batch_size, 128, -1).mean(dim=2)
        x_val = F.relu(self.fc1(x_val))
        value = torch.tanh(self.value_head(x_val))
        return value, logits_policy

# =====================================================================
# 2. HÀM CHUYỂN ĐỔI SANG ONNX
# =====================================================================
def convert_to_onnx(pth_path, onnx_path):
    print("============================================================")
    print("🚀 XUẤT MÔ HÌNH PYTORCH SANG ONNX CHO RUST / OPENVINO")
    print(f"📦 File Checkpoint : {pth_path}")
    print(f"💾 File ONNX đích  : {onnx_path}")
    print("============================================================\n")

    device = torch.device("cpu")
    model = XiangqiNet(num_res_blocks=8, channels=128)

    if os.path.exists(pth_path):
        checkpoint = torch.load(pth_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print("✅ Đã nạp trọng số thành công từ Checkpoint!")
    else:
        print(f"⚠️ Cảnh báo: Không tìm thấy file {pth_path}! Xuất model với trọng số ngẫu nhiên.")

    model.eval()

    # Dữ liệu giả lập input: [batch_size=1, channels=14, height=10, width=9]
    dummy_input = torch.randn(1, 14, 10, 9, dtype=torch.float32)

    # Xuất mô hình ONNX với Dynamic Batch Size (Khớp 100% với OpenVINO Rust)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['value', 'policy'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'value': {0: 'batch_size'},
            'policy': {0: 'batch_size'}
        }
    )
    print(f"🎉 Xuất file ONNX thành công: {onnx_path}")

    # Kiểm tra thử bằng onnxruntime (nếu có cài)
    try:
        import onnxruntime as ort
        with torch.no_grad():
            py_v, py_p = model(dummy_input)

        ort_sess = ort.InferenceSession(onnx_path)
        outputs = ort_sess.run(None, {'input': dummy_input.numpy()})

        diff_v = np.max(np.abs(py_v.numpy() - outputs[0]))
        diff_p = np.max(np.abs(py_p.numpy() - outputs[1]))

        print(f"\n🔍 KIỂM THỬ ĐỘ LỆCH INFERENCE:")
        print(f"   ↳ Độ lệch Value  max: {diff_v:.2e}")
        print(f"   ↳ Độ lệch Policy max: {diff_p:.2e}")
        if diff_v < 1e-4 and diff_p < 1e-4:
            print("🎯 HOÀN HẢO: Output của PyTorch và ONNX khớp nhau 100%!")
    except ImportError:
        print("ℹ️ onnxruntime chưa cài đặt (bỏ qua bước kiểm thử so sánh).")

if __name__ == "__main__":
    pth_input = sys.argv[1] if len(sys.argv) > 1 else "./wisdom_net_v1.pth"
    onnx_output = sys.argv[2] if len(sys.argv) > 2 else "./wisdom_net_v1.onnx"
    convert_to_onnx(pth_input, onnx_output)
