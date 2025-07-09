import torch
print(torch.cuda.is_available())  # 如果返回 True，说明支持 CUDA
print(torch.version.cuda)