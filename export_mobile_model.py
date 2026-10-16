import os
import sys
import torch

sys.stdout.reconfigure(encoding='utf-8')
from app import ForensicNeuralEngine

print("Initializing ForensicNeuralEngine on CPU for mobile export...")
model = ForensicNeuralEngine().cpu()
model.eval()

# Dummy input tensors for MobileNetV3 (RGB 224x224) and SRM/FFT (Grayscale 512x512)
dummy_rgb = torch.randn(1, 3, 224, 224, dtype=torch.float32)
dummy_gray = torch.randn(1, 1, 512, 512, dtype=torch.float32)

os.makedirs('mobile_export', exist_ok=True)

# 1. Export to TorchScript (Mobile-optimized PyTorch)
print("\n--- 1. EXPORTING TO TORCHSCRIPT (MOBILE .PT) ---")
try:
    scripted_model = torch.jit.trace(model, (dummy_rgb, dummy_gray))
    ts_path = os.path.join('mobile_export', 'pixelguard_engine.pt')
    scripted_model.save(ts_path)
    file_size_mb = os.path.getsize(ts_path) / (1024 * 1024)
    print(f"✅ Successfully exported TorchScript Mobile model: {ts_path} ({file_size_mb:.2f} MB)")
    
    # Test loading and inference
    loaded = torch.jit.load(ts_path)
    out = loaded(dummy_rgb, dummy_gray)
    print("✅ Verified TorchScript Mobile forward pass output shape:", out[0].shape)
except Exception as e:
    print("❌ TorchScript export error:", e)

# 2. Export to ONNX (for onnxruntime / TFLite)
print("\n--- 2. EXPORTING TO ONNX ---")
try:
    onnx_path = os.path.join('mobile_export', 'pixelguard_engine.onnx')
    torch.onnx.export(
        model,
        (dummy_rgb, dummy_gray),
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['rgb_input', 'gray_input'],
        output_names=['prob_auth', 'srm_res', 'fft_mag'],
        dynamic_axes={
            'rgb_input': {0: 'batch_size'},
            'gray_input': {0: 'batch_size'},
            'prob_auth': {0: 'batch_size'},
        }
    )
    file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"✅ Successfully exported ONNX model: {onnx_path} ({file_size_mb:.2f} MB)")
    
    # Test inference using onnxruntime
    import onnxruntime as ort
    session = ort.InferenceSession(onnx_path)
    inputs = {
        'rgb_input': dummy_rgb.numpy(),
        'gray_input': dummy_gray.numpy()
    }
    outputs = session.run(None, inputs)
    print("✅ Verified ONNX Runtime inference! Output probability shape:", outputs[0].shape)
except Exception as e:
    print("❌ ONNX export error:", e)

print("\n--- SUMMARY OF EXPORTED ARTIFACTS ---")
for f in os.listdir('mobile_export'):
    p = os.path.join('mobile_export', f)
    print(f"📦 {f}: {os.path.getsize(p)/(1024*1024):.2f} MB")
