from ultralytics import YOLO
import torch
from ultralytics.utils.loss import BboxLoss
import inspect
print(inspect.signature(BboxLoss.forward))

# build model from yaml
model = YOLO("ultralytics/cfg/models/11/yolo11-p2p.yaml")
model.info()  # prints layers, params, GFLOPs — should be ~2M params

# dummy forward pass
x = torch.randn(1, 3, 1280, 1280)
y = model.model(x)


model.train(data="dataset/data.yaml", epochs=1)