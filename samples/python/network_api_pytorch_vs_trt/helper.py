import os
import multiprocessing
if (multiprocessing.cpu_count()>= 40):
    import subprocess
    dev = '1' if "Xeon(R) Gold 6133" in subprocess.run(['lscpu'], capture_output=True, text=True).stdout else '3'
else:
    dev = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = dev
# CUDA_VISIBLE_DEVICES set before importing torch

import torch
import resource
import time

class nvtxTimeTable(object):
    def __init__(self, allTimeStr=None, use_nvtx=False):
        '''
        初始化时给出总时间的字符串, 占据首位; 默认关闭 use_nvtx, 避免影响性能
        '''
        self.timeDict = {}
        if allTimeStr is not None:
            self.timeDict[allTimeStr] = 0
        self.nvtxFlag = use_nvtx
        self.startTimeStack = []
        self.strStack = []
        self.strID = 0
        self.MultiEpochTime = {}

    def start(self):
        return time.perf_counter()

    def elapsed(self, start):
        return time.perf_counter() - start
        
    def nvtx_push(self,str="nvtxTag"):
        if self.nvtxFlag:
            torch.cuda.nvtx.range_push(str)

    def nvtx_pop(self):
        if self.nvtxFlag:
            torch.cuda.nvtx.range_pop()

    def nvtx_start(self):
        '''
        nsys分析时, 搭配选项 --capture-range=cudaProfilerApi后, 需要调用start
        '''
        if self.nvtxFlag == 1: 
            torch.cuda.cudart().cudaProfilerStart()

    def nvtx_stop(self):
        self.strID = 0
        if self.nvtxFlag == 1: 
            torch.cuda.cudart().cudaProfilerStop()

    def time_push(self, keyStr=None):
        '''
        push the start time to the stack, work with nvtx_push
        '''
        # config str
        if keyStr == None:
            self.strID += 1
            keyStr = "t_" + str(self.strID)
        # push
        self.strStack.append(keyStr)
        self.startTimeStack.append(self.start())
        if self.nvtxFlag:
            self.nvtx_push(keyStr)
        if keyStr not in self.timeDict.keys():
            self.timeDict[keyStr] = 0
        return self.start()
    
    def time_pop(self, NoUseStr=None):
        '''
        pop the start time from the stack, work with nvtx_push
        '''
        if self.strStack:
            keyStr = self.strStack.pop()
        else:
            raise IndexError("pop from empty strStack")
        if self.startTimeStack:
            start = self.startTimeStack.pop()
        else:
            raise IndexError("pop from empty startTimeStack")
        if self.nvtxFlag:
            self.nvtx_pop()
        # save
        self.timeDict[keyStr] += self.elapsed(start)
        return self.elapsed(start)
    
    def save_print_time(self, pFlag=False):
        '''
        每轮结束时调用，打印，保存，刷新
        '''
        result = "== " # "\t"
        result += ", ".join(self.timeDict.keys()) + ": "
        result += ", ".join(f"{v:.4f}" for v in self.timeDict.values())
        if pFlag: print(result)
        if not self.MultiEpochTime: # 第一次保存时，初始化
            for k in self.timeDict.keys():
                self.MultiEpochTime[k] = [] # 用列表 append 方便丢弃预热轮
        
        for k in self.timeDict.keys():
            self.MultiEpochTime[k].append(self.timeDict[k])
            self.timeDict[k] = 0
        
        return result

    def print_avg_stats(self, dropWarmNum=0):
        ''' 
        calculate averages, skip N points because of warmup
        '''
        avg_stats = {}
        for k, v in self.MultiEpochTime.items():
            if len(v) > dropWarmNum:
                valid_data = v[dropWarmNum:]
                avg_stats[k] = sum(valid_data) / len(valid_data) if valid_data else 0
            else:
                avg_stats[k] = sum(v) / len(v) if v else 0
        
        # print(f'=== Epoch Num: {len(self.MultiEpochTime[k])}, Drop Warmup Epoch Num: {dropWarmNum}')
        print("=== AVG : ",end="")
        print(", ".join(avg_stats.keys()),end=": ")
        print(", ".join(f"{v:.2f}" for v in avg_stats.values()))

nvTT = nvtxTimeTable("t_allTime", True) # 确保首时间第一个定义

def PRINT_PEAK_MEM(DEV_MEM=0):
    # Print peak memory usage for both host and device
    hostPeakMemMB = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    if DEV_MEM == 0:
        DEV_MEM = (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else 0

    print(f"=== Peak Host Memory, Peak Device Memory:(GB) { hostPeakMemMB :.2f}, {DEV_MEM:.2f}")
    return hostPeakMemMB, DEV_MEM

import torch
import torch.nn as nn
import gc
from collections import defaultdict
# import GPUtil

class MemoryProfiler:
    def __init__(self, model, use_torch_mem=True):
        self.model = model
        self.memory_log = []
        self.tensor_ids = set()
        self.recordFlag = True
        self.peak_mem = 0
        self.itr = 0
        self.maXitr = 0
        self.GPUs  = None
        self.ONLY_TORCH_TENSOR=use_torch_mem

    def _get_tensor_memory(self, tensor):
        """获取单个tensor的显存占用"""
        if tensor is None:
            return 0
        if not isinstance(tensor, torch.Tensor):
            return 0
        if not tensor.is_cuda:
            return 0
        
        # 计算tensor占用的字节数
        numel = tensor.numel()
        element_size = tensor.element_size()
        return numel * element_size / 1024**2  # 转换为MB
    
    def _record_memory_snapshot(self, step_name, tensors=None, printFlag = False):
        """记录显存快照, 入口名称建议 afOP, 即表示 OP 完成后的显存状态"""
        if not self.recordFlag:
            return
        if self.ONLY_TORCH_TENSOR:
            snapshot = {
                'step': step_name,
                'allocated': torch.cuda.memory_allocated() / 1024**2,
                'reserved': torch.cuda.memory_reserved() / 1024**2,
                'max_allocated': torch.cuda.max_memory_allocated() / 1024**2,
                'tensors': defaultdict(float)
            }
        else:
            import pynvml as ml
            ml.nvmlInit()
            handle = ml.nvmlDeviceGetHandleByIndex(3) #
            info = ml.nvmlDeviceGetMemoryInfo(handle)

            snapshot = {
                'step': step_name,
                'allocated': info.used / 1024**2,
                'reserved': info.used / 1024**2, # TODO: fix
                'max_allocated': info.free / 1024**2,# TODO: fix
                'tensors': defaultdict(float)
            }
        
        # 如果有tensor列表，记录每个tensor的显存
        if tensors is not None:
            for name, tensor in tensors.items():
                mem = self._get_tensor_memory(tensor)
                if mem > 0:
                    snapshot['tensors'][name] = {
                        'memory_mb': mem,
                        'shape': tensor.shape if hasattr(tensor, 'shape') else None,
                        'dtype': str(tensor.dtype) if hasattr(tensor, 'dtype') else None
                    }
        
        self.memory_log.append(snapshot)
        if printFlag:
            print(f"[{step_name}]",end=': ')
            print(f"  Allocated: {snapshot['allocated']:.2f} MB",end='; ')
            print(f"  Reserved: {snapshot['reserved']:.2f} MB",end='; ')
            print(f"  Peak: {snapshot['max_allocated']:.2f} MB")
            
            if any(v['memory_mb'] > 0 for v in snapshot['tensors'].values()):
                for name, info in snapshot['tensors'].items():
                    if info['memory_mb'] > 0:
                        shape_str = str(info['shape']) if info['shape'] is not None else "None"
                        dtype_str = info['dtype'] if info['dtype'] is not None else "Unknown"
                        print(f"    {name}: {info['memory_mb']:.2f} MB {shape_str} Dtype: {dtype_str}")
        
        return snapshot
    
    def print_summary(self):
        """打印显存使用总结"""
        print("\n" + "="*50)
        print("Memory Usage Summary:")
        print("="*50)
        
        for i, log in enumerate(self.memory_log):
            if i > 0:
                prev = self.memory_log[i-1]
                delta = log['allocated'] - prev['allocated']
                sign = "+" if delta >= 0 else ""
                print(f"{log['step']:20s}: {log['allocated']:8.4f} MB ({sign}{delta:6.2f} MB)")
            else:
                print(f"{log['step']:20s}: {log['allocated']:8.4f} MB")

    def print_peak_memory(self, ReturnnoPrint=0):
        """打印峰值显存使用量"""
        if ReturnnoPrint:
            return self.peak_mem
        else:
            print(f"Peak Memory Usage [{self.maXitr}]: {self.peak_mem:.2f} MB")

    def reset(self,gpu_num=0, reset_peak=False):
        """重置显存快照记录, 需要时重置显存峰值"""
        
        self.itr += 1
        # 记录当前的峰值（重置前）
        tmp = 0
        if 1: # not accurate
            if self.memory_log:
                tmp = self.memory_log[-1]['max_allocated']
        if self.recordFlag:# accurate but performance harmfully
            # if(0 or self.itr%30==0): 
            #     self.GPUs = GPUtil.getGPUs() # TODO :准确统计显存峰值，但严重影响性能
            #     tmp = self.GPUs[gpu_num].memoryUsed
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        if tmp > self.peak_mem:
            self.peak_mem = tmp
            self.maXitr = self.itr

        self.memory_log = []
        
        # 如果需要重置峰值记录
        if reset_peak:
            self.peak_mem = 0

    def monitor_forward(self, *args, **kwargs):
        """EXAMPLE: 手动trace前向传播的样例"""
        # 清理显存
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # 记录初始状态
        self._record_memory_snapshot("Initial")
        
        # 获取输入
        x = args[0]
        self._record_memory_snapshot("Input", {'input': x})
        
        # 方法1：使用模型的forward方法，但手动记录中间步骤
        # 这需要知道模型的具体结构
        # with torch.no_grad():
        if 1:
            # conv1
            x = self.model.conv1(x)
            self._record_memory_snapshot("After conv1", {'output_conv1': x})
            
            # bn1
            x = self.model.bn1(x)
            self._record_memory_snapshot("After bn1", {'output_bn1': x})
            
            # relu
            x = self.model.relu(x)
            self._record_memory_snapshot("After relu", {'output_relu': x})
            
            # pool
            x = self.model.pool(x)
            self._record_memory_snapshot("After pool", {'output_pool': x})
            
            # conv2
            x = self.model.conv2(x)
            self._record_memory_snapshot("After conv2", {'output_conv2': x})
            
            # bn2
            x = self.model.bn2(x)
            self._record_memory_snapshot("After bn2", {'output_bn2': x})
            
            # relu
            x = self.model.relu(x)
            self._record_memory_snapshot("After relu2", {'output_relu2': x})
            
            # pool
            x = self.model.pool(x)
            self._record_memory_snapshot("After pool2", {'output_pool2': x})
            
            # flatten
            x = x.flatten(1)
            self._record_memory_snapshot("After flatten", {'output_flatten': x})
            
            # fc
            x = self.model.fc(x)
            self._record_memory_snapshot("After fc", {'output_fc': x})
            
            gc.collect()
            torch.cuda.empty_cache()
        
        return x

    def monitor_forward_auto(self, *args, **kwargs):
        """TODO: 自动监控前向传播（适用于任意模型）"""
        # 清理显存
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # 记录初始状态
        self._record_memory_snapshot("Initial")
        
        # 获取输入
        x = args[0]
        self._record_memory_snapshot("Input", {'input': x})
        
        # 注册hook来监控每个模块的输出
        hooks = []
        
        def make_hook(name):
            def hook(module, input, output):
                self._record_memory_snapshot(f"After {name}", {f'output_{name}': output})
                return output
            return hook
        
        # 为每个子模块注册hook
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # 只给叶子模块注册
                hook = module.register_forward_hook(make_hook(name))
                hooks.append(hook)
        
        # 执行前向传播
        # with torch.no_grad():
        output = self.model(x)
        
        # 移除所有hook
        for hook in hooks:
            hook.remove()
        
        # 记录最终输出
        self._record_memory_snapshot("Final output", {'output': output})
        
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        return output

    
MP = MemoryProfiler(model=None)  # 传入模型实例

if __name__ == "__main__":
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(2)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(128)
            # 注意：计算正确的全连接层输入维度
            # 输入: 32x32, 经过两次pool变成8x8
            # 所以最终特征图大小: batch_size * 128 * 8 * 8
            self.fc = nn.Linear(128 * 8 * 8, 10)
            
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.pool(x)
            
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.pool(x)
            
            x = x.flatten(1)
            x = self.fc(x)
            return x
    
    # 创建模型和输入
    model = SimpleModel().cuda()
    input_tensor = torch.randn(32, 3, 32, 32).cuda()
    
    # 方法1：使用手动监控（需要知道模型结构）
    print("=== Method 1: Manual Monitoring ===")
    profiler = MemoryProfiler(model)
    output = profiler.monitor_forward(input_tensor)
    profiler.print_summary()
    
    # 方法2：使用自动hook监控（通用方法）
    print("\n" + "="*50)
    print("=== Method 2: Automatic Hook Monitoring ===")
    profiler2 = MemoryProfiler(model)
    output2 = profiler2.monitor_forward_auto(input_tensor)
    profiler2.print_summary()