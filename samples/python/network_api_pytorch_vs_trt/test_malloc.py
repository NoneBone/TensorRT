import numpy as np
from cuda.bindings import driver as cuda, runtime as cudart

nbytes = 1024

# 1. 分配页锁定主机内存
ret, host_ptr = cudart.cudaMallocHost(nbytes)
if ret != cudart.cudaError_t.cudaSuccess:
    raise RuntimeError(f"cudaMallocHost failed: {ret}")

print("✅ cudaMallocHost success")

# 2. 把指针包装成 NumPy 缓冲区
buf = np.core.multiarray.int_asbuffer(int(host_ptr), nbytes)
arr = np.frombuffer(buf, dtype=np.uint8)

# 3. 写入示例数据
arr[:4] = [1, 2, 3, 4]
print("前 10 bytes:", arr[:10])

# 4. 释放
cudart.cudaFreeHost(host_ptr)
# import torch

# def test_cuda_malloc():
#     print("PyTorch version:", torch.__version__)
#     print("CUDA available:", torch.cuda.is_available())
    
#     if not torch.cuda.is_available():
#         print("❌ CUDA 不可用，终止测试")
#         return

#     device = torch.device("cuda:0")
#     print("Device:", torch.cuda.get_device_name(device))

#     try:
#         # 分配 100MB 显存
#         size = 100 * 1024 * 1024  # 100MB
#         tensor = torch.empty(size, dtype=torch.uint8, device=device)
#         print(f"✅ 成功分配 {size / 1024 / 1024:.1f} MB 显存")

#         # 简单写一下，防止被优化掉
#         tensor[0] = 1

#         # 释放
#         del tensor
#         torch.cuda.empty_cache()
#         print("✅ 显存已释放")

#     except RuntimeError as e:
#         print("❌ CUDA 运行时错误:")
#         print(e)

# if __name__ == "__main__":
#     test_cuda_malloc()