'''

在使用 sample 得到 onnx 模型文件后，开始构建 engine 文件，以完成推理
# onnx fastly verify
'''

import numpy as np
import onnxruntime as ort
sess = ort.InferenceSession("mnist_fp32_dBS.onnx")
dummy = np.random.randn(32, 1, 28, 28).astype(np.float32)   # batch=32
output = sess.run(None, {"input": dummy})
print(output[0].shape)