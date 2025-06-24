# 🔊 实时语音识别系统（基于 Sherpa-ONNX）

本项目实现了一个基于 **Sherpa-ONNX** 的实时语音识别系统，结合 **VAD（Voice Activity Detection）** 和 **FireRed ASR** 模型，实现了从麦克风输入到语音识别输出的完整流程。

---

## 🔧 主要功能

- **🎙 实时语音识别**
  - 使用 `sounddevice` 实时采集麦克风音频数据。
  - 基于 FireRed ASR 模型进行非流式语音识别，支持中英文混合。

- **🧩 语音活动检测（VAD）**
  - 使用 Silero VAD 模型检测语音起止点，实现语音端点检测。
  - 支持自动截断静音部分，提升识别效率。

- **🔥 热词匹配与替换**
  - 加载用户定义的热词及其拼音规则。
  - 使用模糊匹配算法（FuzzyWuzzy）将识别结果中的错误发音替换为预设热词。

- **🪄 文本后处理**
  - 加载自定义文本替换规则，对识别结果进行规范化处理。
  - 使用 `pypinyin` 将中文转换为拼音，辅助热词匹配。

- **✒️ 标点恢复**
  - 集成 `sherpa_onnx.OfflinePunctuation` 模块，为识别结果添加标点符号，提高可读性。

- **⚙️ 命令行参数配置**
  - 支持灵活配置模型路径、线程数、热词文件、替换规则等参数。

- **🧵 多线程架构**
  - 使用 `threading` 实现录音与识别的并行处理，提升响应速度和系统稳定性。

---

## 🧠 使用的模型与工具

| 类别           | 名称                                                                 |
|----------------|----------------------------------------------------------------------|
| 语音识别模型   | FireRed ASR（encoder + decoder，ONNX 格式）                         |
| 语音活动检测   | Silero VAD 模型（ONNX 格式）                                        |
| 标点恢复       | sherpa-onnx-punct-ct-transformer 模型                                |
| 音频采集       | [sounddevice](https://pypi.org/project/sounddevice/)                 |
| 热词匹配       | [fuzzywuzzy](https://github.com/seatgeek/fuzzywuzzy)                |
| 拼音转换       | [pypinyin](https://github.com/mozillazg/python-pinyin)              |

---

## 🚀 启动方式示例

```bash
python3 simulate-streaming-fire-red-microphone.py \
    --silero-vad-model=./model/sense-voice/silero_vad.onnx \
    --tokens=./model/sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/tokens.txt \
    --fire-red-asr-encoder=./model/sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/encoder.int8.onnx \
    --fire-red-asr-decoder=./model/sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/decoder.int8.onnx \
    --hotwords-file=./file/hotwords.txt \
    --replace-rules=./file/replace_rules.txt
