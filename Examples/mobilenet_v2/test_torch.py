import torch, torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
# import torchvision.datasets as datasets
import torch.utils.data as data
# import torchvision.transforms as transforms
from torch.autograd import Variable
# import torchvision.models as models
# import matplotlib.pyplot as plt
import time, os, copy, numpy as np
from dataset import get_dataset

# model_path = "/mnt/share_disk/bruce_trie/workspace/Quantizer-Tools/_outputs/models/mobile_v2_best_model_200_labels.pth"
# model_path = "/mnt/share_disk/bruce_trie/workspace/Quantizer-Tools/MQBench/Examples/models/mbv2_fp16.pth"
# model_path = "/mnt/share_disk/bruce_trie/workspace/Quantizer-Tools/MQBench/Examples/models/mobilenet_v2-b0353104.pth"
model_path = "/mnt/share_disk/bruce_trie/workspace/Quantizer-Tools/MQBench/Examples/models/mobilenet_v2.pth"
model = torch.load(model_path)

_, val_dataset, _ = get_dataset()
dataloaders = torch.utils.data.DataLoader(val_dataset, batch_size=512, shuffle=True, num_workers=8)

running_corrects = 0.0
for i, (inputs, labels) in enumerate(dataloaders):
    inputs = inputs.cuda()
    labels = labels.cuda()
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    running_corrects += torch.sum(preds == labels.data)
print(f"Accuracy : {running_corrects / len(val_dataset) * 100}%")

# convert to onnx
if isinstance(model, torch.nn.DataParallel):
    model = model.module
x = torch.randn(1, 3, 224, 224).cuda()

onnx_path = "/mnt/share_disk/bruce_trie/workspace/Quantizer-Tools/_outputs/models/mobilev2_model_qat.onnx"
torch.onnx.export(
    model, 
    x, 
   onnx_path, 
   export_params=True, 
   opset_version=11
)