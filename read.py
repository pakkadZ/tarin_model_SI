import torch
from torchvision.ops import nms

boxes = torch.tensor([[0, 0, 10, 10], [0, 0, 9, 9]], dtype=torch.float32).cuda()
scores = torch.tensor([0.9, 0.8], dtype=torch.float32).cuda()

print(nms(boxes, scores, 0.5))