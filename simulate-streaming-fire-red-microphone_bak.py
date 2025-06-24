#!/usr/bin/env python3
#
# Copyright (c)  2025  Xiaomi Corporation

"""
This file demonstrates how to use sherpa-onnx Python APIs
with VAD and non-streaming SenseVoice for real-time speech recognition
from a microphone.

Usage:


wget https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx

./python-api-examples/simulate-streaming-sense-voice-microphone.py  \
  --silero-vad-model=./silero_vad.onnx \
  --sense-voice=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.onnx \
  --tokens=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt
"""
import argparse
import os
import queue
import sys
import threading
import time
from asyncio import Queue
from pathlib import Path

import numpy as np
from fuzzywuzzy import fuzz
from pypinyin import lazy_pinyin, Style

# 全局队列，用于前后端通信
result_queue = Queue()

启动命令= """
python3 simulate-streaming-fire-red-microphone.py 
    --silero-vad-model=./model/sense-voice/silero_vad.onnx
    --tokens=./model/sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/tokens.txt
    --fire-red-asr-encoder=./model/sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/encoder.int8.onnx
    --fire-red-asr-decoder=./model/sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/decoder.int8.onnx
    --hotwords-file=./file/hotwords.txt 
    --replace-rules=./file/replace_rules.txt
  """

try:
    import sounddevice as sd
except ImportError:
    print("Please install sounddevice first. You can use")
    print()
    print("  pip install sounddevice")
    print()
    print("to install it")
    sys.exit(-1)

import sherpa_onnx

killed = False
recording_thread = None
sample_rate = 16000  # Please don't change it

# buffer saves audio samples to be played
samples_queue = queue.Queue()

class HotwordMatcher:
    def __init__(self, hotwords_file=None, threshold=75):
        self.hotwords = []
        self.threshold = threshold
        if hotwords_file:
            self.load_hotwords(hotwords_file)

    def load_hotwords(self, hotwords_file):
        """加载热词及其拼音"""
        self.hotwords = []
        with open(hotwords_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) < 2:
                    continue
                word, pinyin = parts[0], parts[1].split()
                self.hotwords.append((word, pinyin))

    def match(self, text):
        if not self.hotwords or not text:
            return text

        # 在匹配前先转换中文数字

        text_pinyin = lazy_pinyin(text, style=Style.NORMAL)
        matched = False
        new_text = list(text)  # 转换为字符列表以便替换
        # 按热词长度从长到短排序，优先替换长词
        sorted_hotwords = sorted(self.hotwords, key=lambda x: len(x[1]), reverse=True)

        for i in range(len(text_pinyin)):
            for word, hotword_pinyin in sorted_hotwords:
                length = len(hotword_pinyin)
                if i + length > len(text_pinyin):
                    continue

                sub_pinyin = text_pinyin[i:i + length]
                scores = [fuzz.ratio(tp, hp) for tp, hp in zip(sub_pinyin, hotword_pinyin)]
                if all(score > 90 for score in scores) and sum(scores) / len(scores) > 95:
                    # 替换字符列表中的对应位置
                    new_text[i:i + length] = list(word)
                    matched = True
                    i += length - 1  # 跳过已替换的字符
                    break

        text = ''.join(new_text) if matched else text
        return text

# 加载文本替换规则
def load_replacement_rules(file_path):
    rules = []
    if not file_path or not os.path.exists(file_path):
        return rules

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "->" not in line:
                continue
            original, replacement = line.split("->", 1)
            original = original.strip()
            replacement = replacement.strip()
            if original and replacement:
                rules.append((original, replacement))

    # 按原始文本长度从长到短排序，确保长匹配优先
    rules.sort(key=lambda x: len(x[0]), reverse=True)
    return rules

# 文本替换函数
def apply_replacement_rules(text, rules):
    for original, replacement in rules:
        if original in text:
            print("替换：", original, "->", replacement)
            text = text.replace(original, replacement)
    return text


"""
该函数定义了命令行参数解析器，用于配置语音识别系统相关参数：
热词文件(--hotwords-file)：指定热词及其拼音映射
替换规则(--replace-rules)：定义文本替换格式
模型路径参数：指定VAD模型(--silero-vad-model)、token文件(--tokens)、SenseVoice编解码模型路径
计算配置：线程数(--num-threads，默认2)
同音词替换参数：指定jieba字典目录(--hr-dict-dir)、词典文件(--hr-lexicon)和替换规则FST(--hr-rule-fsts)
"""
def get_args():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--hotwords-file",
        type=str,
        default="",
        help="""
            包含热词的文件，每行格式为 "热词 拼音"，例如：
            HELLO WORLD hello world
            你好世界 ni hao shi jie
            """
    )

    parser.add_argument(
        "--replace-rules",
        type=str,
        default="",
        help="包含文本替换规则的文件的路径（格式：'original -> replacement'）"
    )

    parser.add_argument(
        "--silero-vad-model",
        type=str,
        required=True,
        help="Path to silero_vad.onnx",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        help="Path to tokens.txt",
    )

    parser.add_argument(
        "--fire-red-asr-encoder",
        default="",
        type=str,
        help="Path to the model.onnx from SenseVoice",
    )

    parser.add_argument(
        "--fire-red-asr-decoder",
        default="",
        type=str,
        help="Path to the model.onnx from SenseVoice",
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=2,
        help="Number of threads for neural network computation",
    )

    parser.add_argument(
        "--hr-dict-dir",
        type=str,
        default="",
        help="If not empty, it is the jieba dict directory for homophone replacer",
    )

    parser.add_argument(
        "--hr-lexicon",
        type=str,
        default="",
        help="If not empty, it is the lexicon.txt for homophone replacer",
    )

    parser.add_argument(
        "--hr-rule-fsts",
        type=str,
        default="",
        help="If not empty, it is the replace.fst for homophone replacer",
    )

    return parser.parse_args()


def assert_file_exists(filename: str):
    assert Path(filename).is_file(), (
        f"{filename} does not exist!\n"
        "Please refer to "
        "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html to download it"
    )


"""
该函数创建一个基于FireRed ASR模型的离线语音识别器，功能如下：
使用指定的编码器(encoder)和解码器(decoder)模型文件
加载token列表文件进行文本解码
支持多线程推理(由num_threads控制)
配置热词优化(hotword)相关参数：
调试模式(debug)默认关闭
当前使用FireRed ASR模型实现。
"""
def create_recognizer(args) -> sherpa_onnx.OfflineRecognizer:
    # assert_file_exists(args.sense_voice)
    # recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
    #     model=args.sense_voice,
    #     tokens=args.tokens,
    #     num_threads=args.num_threads,
    #     use_itn=False,
    #     debug=False,
    #     hr_dict_dir=args.hr_dict_dir,
    #     hr_rule_fsts=args.hr_rule_fsts,
    #     hr_lexicon=args.hr_lexicon,
    # )
    recognizer = sherpa_onnx.OfflineRecognizer.from_fire_red_asr(
        encoder=args.fire_red_asr_encoder,
        decoder=args.fire_red_asr_decoder,
        tokens=args.tokens,
        num_threads=args.num_threads,
        debug=False,
        hr_dict_dir=args.hr_dict_dir,
        hr_rule_fsts=args.hr_rule_fsts,
        hr_lexicon=args.hr_lexicon,
    )

    return recognizer


"""
该函数实现持续录音并缓存数据，具体流程如下：
设置每次读取时长为100ms（基于采样率计算）
创建单声道浮点型音频输入流
循环执行：
阻塞式读取音频数据块
将二维数据展平为一维数组
拷贝数据副本防止内存泄漏
将音频数据放入队列缓冲区
通过全局变量killed控制循环终止
注：依赖外部定义的sample_rate(采样率)、killed(终止标志)和samples_queue(数据队列)三个变量。
"""
def start_recording():
    # You can use any value you like for samples_per_read
    samples_per_read = int(0.1 * sample_rate)  # 0.1 second = 100 ms

    with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
        while not killed:
            samples, _ = s.read(samples_per_read)  # a blocking read
            samples = samples.reshape(-1)
            samples = np.copy(samples)
            samples_queue.put(samples)


def update_frontend(text):
    print("识别结果:", text)
    result_queue.put(text)

"""
该代码实现了一个实时语音识别系统，主要功能如下：
设备检测：检查麦克风设备并选择默认输入设备
参数处理：解析命令行参数（模型路径/替换规则/线程数等）
模型加载：
加载语音识别模型(FireRed-ASR)
加载VAD语音活动检测模型
加载标点恢复模型
实时录音：启动录音线程持续采集音频数据
语音检测：使用VAD检测语音活动并分割音频段
语音识别：
对音频段进行流式识别
支持热词匹配替换（如"HELLO WORLD"→"hello world"）
后处理：
添加标点符号
应用文本替换规则（如自定义缩写展开）
结果展示：实时显示识别结果并支持句子结束标记
通过多线程实现录音与识别的并行处理，结合VAD实现语音端点检测，最终输出带标点的规范化文本。
"""
def main():
    devices = sd.query_devices()
    if len(devices) == 0:
        print("No microphone devices found")
        sys.exit(0)

    print(devices)

    # If you want to select a different input device, please use
    # sd.default.device[0] = xxx
    # where xxx is the device number

    default_input_device_idx = sd.default.device[0]
    print(f'Use default device: {devices[default_input_device_idx]["name"]}')

    args = get_args()
    print("Received arguments:", args)
    assert_file_exists(args.tokens)
    assert_file_exists(args.silero_vad_model)

    assert args.num_threads > 0, args.num_threads

    print("Creating recognizer. Please wait...")
    recognizer = create_recognizer(args)

    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = args.silero_vad_model
    config.silero_vad.threshold = 0.5
    config.silero_vad.min_silence_duration = 0.1  # seconds
    config.silero_vad.min_speech_duration = 0.25  # seconds
    # If the current segment is larger than this value, then it increases
    # the threshold to 0.9 internally. After detecting this segment,
    # it resets the threshold to its original value.
    config.silero_vad.max_speech_duration = 8  # seconds
    config.sample_rate = sample_rate

    window_size = config.silero_vad.window_size

    print(f"window_size:{window_size}")

    # 初始化热词匹配器
    hotword_matcher = None
    if args.hotwords_file:
        # 热词匹配器初始化
        hotword_matcher = HotwordMatcher(args.hotwords_file, threshold=75)

    # 初始化文本替换规则
    rules = []
    if args.replace_rules:
        print(f"Loading replacement rules from {args.replace_rules}")
        rules = load_replacement_rules(args.replace_rules)

    vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=85)

    print("Started! Please speak")

    buffer = []

    global recording_thread
    recording_thread = threading.Thread(target=start_recording)
    recording_thread.start()

    display = sherpa_onnx.Display()

    started = False
    started_time = None

    offset = 0

    # ***************
    # 新增：加载标点模型
    # ***************
    punct_model_path = "model/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx"
    if not Path(punct_model_path).is_file():
        raise FileNotFoundError(f"标点模型不存在: {punct_model_path}")

    punct_config = sherpa_onnx.OfflinePunctuationConfig(
        model=sherpa_onnx.OfflinePunctuationModelConfig(ct_transformer=punct_model_path),
    )
    punct = sherpa_onnx.OfflinePunctuation(punct_config)

    while not killed:
        samples = samples_queue.get()  # a blocking read

        buffer = np.concatenate([buffer, samples])
        while offset + window_size < len(buffer):
            vad.accept_waveform(buffer[offset : offset + window_size])
            if not started and vad.is_speech_detected():
                started = True
                started_time = time.time()
            offset += window_size

        if not started:
            if len(buffer) > 10 * window_size:
                offset -= len(buffer) - 10 * window_size
                buffer = buffer[-10 * window_size :]
        text_with_punct = None
        # if started and time.time() - started_time > 0.2:
        #     stream = recognizer.create_stream()
        #     stream.accept_waveform(sample_rate, buffer)
        #     recognizer.decode_stream(stream)
        #     text = stream.result.text.strip()
        #     # 热词匹配后处理
        #     if hotword_matcher and text:
        #         matched_text = hotword_matcher.match(text)
        #         if matched_text != text:
        #             print(f"Hotword matched: {text} → {matched_text}")
        #             text = matched_text
        #
        #     # ***************
        #     # 新增：添加标点
        #     # ***************
        #     if text:
        #         text_with_punct = punct.add_punctuation(text)
        #
        #         # 应用替换规则
        #         text_with_punct = apply_replacement_rules(text_with_punct, rules)
        #
        #         # update_frontend(text_with_punct)
        #
        #     started_time = time.time()


        while not vad.empty():
            # In general, this while loop is executed only once
            stream = recognizer.create_stream()
            stream.accept_waveform(sample_rate, vad.front.samples)

            vad.pop()
            recognizer.decode_stream(stream)

            text = stream.result.text.strip()
            if hotword_matcher and text:
                matched_text = hotword_matcher.match(text)
                if matched_text != text:
                    print(f"Hotword matched: {text} → {matched_text}")
                    text = matched_text

            # ***************
            # 新增：添加标点
            # ***************
            text_with_punct = punct.add_punctuation(text)
            # 应用替换规则
            text_with_punct = apply_replacement_rules(text_with_punct, rules)

            display.update_text(text_with_punct)

            buffer = []
            offset = 0
            started = False
            started_time = None
        if text_with_punct:
            update_frontend(text_with_punct)



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        killed = True
        if recording_thread:
            recording_thread.join()
        print("\nCaught Ctrl + C. Exiting")
