from ultralytics import YOLO

# Load model architecture from YAML (randomly initialized weights)
model = YOLO("ultralytics/cfg/models/11/yolo11-p2p.yaml")

# (Optional) sanity check
print(model)

# Export to ONNX
model.export(
    format="onnx",
    opset=12,          # commonly 12 or 13 for compatibility
    simplify=True,     # simplifies graph (recommended)
    dynamic=False      # set True if you need variable batch size
)