import onnxruntime
print(onnxruntime.__version__)
print(onnxruntime.get_device() ) # 如果得到的输出结果是GPU，所以按理说是找到了GPU的

ort_session = onnxruntime.InferenceSession("model/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx",
providers=['CUDAExecutionProvider'])
print(ort_session.get_providers())