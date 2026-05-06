# Kiểm tra Cuda có được hỗ trợ không
import torch
print(f"CUDA khả dụng: {torch.cuda.is_available()}")
print(f"Thiết bị: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")