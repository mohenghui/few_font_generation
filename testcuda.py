import torch
import time
from torch import autograd
print(torch.__version__)
print(torch.cuda.is_available())

# torch创建gpu设置两种方法：
# Expected one of cpu, cuda, xpu, mkldnn, opengl, opencl, ideep, hip, msnpu, xla
# 1
device = torch.device("cuda")
print(device)
print(torch.cuda.get_device_name(0))
# 2
num_gpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and num_gpu > 0) else "cpu")
print(device)
print(torch.cuda.get_device_name(0))
