import torch
import torch.nn as nn

# target output size of 5
m = nn.AdaptiveAvgPool1d(5)
input = torch.randn(1, 64, 8)
output = m(input)

print(output.shape)