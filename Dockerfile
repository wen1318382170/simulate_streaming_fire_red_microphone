# ARG CUDA_IMAGE="12.4.0-devel-ubuntu22.04"
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
# FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git build-essential \
    python3 python3-pip gcc wget \
    ocl-icd-opencl-dev opencl-headers clinfo \
    libclblast-dev libopenblas-dev \
    && mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends nvidia-cudnn

COPY tar/cudnn-linux-x86_64-8.9.1.23_cuda11-archive.tar.xz .
RUN tar xvf cudnn-linux-x86_64-8.9.1.23_cuda11-archive.tar.xz --strip-components=1 -C /usr/local/cuda-11.8


ENV CUDA_HOME="/usr/local/cuda-11.8"
ENV PATH="$CUDA_HOME/bin:$PATH"
ENV LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
ENV CUDAToolkit_ROOT_DIR="$CUDA_HOME"
ENV CUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME"
ENV CUDA_TOOLKIT_ROOT="$CUDA_HOME"
ENV CUDA_BIN_PATH="$CUDA_HOME"
ENV CUDA_PATH="$CUDA_HOME"
ENV CUDA_INC_PATH="$CUDA_HOME/targets/x86_64-linux/include"
ENV CFLAGS="-I$CUDA_HOME/targets/x86_64-linux/include"
ENV CUDAToolkit_TARGET_DIR="$CUDA_HOME/targets/x86_64-linux"

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        portaudio19-dev \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY sherpa_onnx-1.12.0%2Bcuda-cp310-cp310-linux_x86_64.whl .
RUN pip install sherpa_onnx-1.12.0%2Bcuda-cp310-cp310-linux_x86_64.whl
# RUN pip install sherpa-onnx==1.12.0+cuda -f https://k2-fsa.github.io/sherpa/onnx/cuda-cn.html
# 复制代码
COPY . .

# 暴露服务端口
EXPOSE 9090

# 启动服务（使用容器内路径）
CMD ["python3", "app.py"]
