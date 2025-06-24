import argparse
import logging
import os
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sherpa_onnx
from fuzzywuzzy import fuzz
from pypinyin import lazy_pinyin, Style

# 全局状态管理
client_states = {}  # 客户户ID -> 状态字典
state_lock = threading.Lock()

# 客户端队列管理
client_queues = {}  # 客户端ID -> 队列
queue_lock = threading.Lock()

# 配置日志（改为DEBUG级别）
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

# 性能指标常量
MAX_LOG_LENGTH = 100  # 最大日志记录长度


def add_client_queue(client_id):
    """创建客户端专属队列"""
    with queue_lock:
        client_queues[client_id] = queue.Queue()
    logging.debug(f"[{client_id}] 客户端队列已创建")


def remove_client_queue(client_id):
    """移除客户端队列"""
    with queue_lock:
        client_queues.pop(client_id, None)
    logging.debug(f"[{client_id}] 客户端队列已移除")


def get_client_queue(client_id):
    """获取客户端队列"""
    with queue_lock:
        return client_queues.get(client_id)
    return None


# 全局模型资源（共享）
global_recognizer = None
global_punct = None
global_vad_model = None
global_vad_window_size = 512

# 全局控制标志
kill_flag = threading.Event()
kill_flag.set()
sample_rate = 16000

# 全局后处理资源
hotword_matcher = None
replacement_rules = []


def initialize_global_resources(args):
    """初始化全局模型资源"""
    global global_recognizer, global_punct, global_vad_model, global_vad_window_size
    global hotword_matcher, replacement_rules

    if global_recognizer is None:
        global_recognizer = create_recognizer(args)

    if global_punct is None:
        punct_model_path = "model/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx"
        if not Path(punct_model_path).is_file():
            raise FileNotFoundError(f"标点模型不存在: {punct_model_path}")
        punct_config = sherpa_onnx.OfflinePunctuationConfig(
            model=sherpa_onnx.OfflinePunctuationModelConfig(ct_transformer=punct_model_path),
        )
        global_punct = sherpa_onnx.OfflinePunctuation(punct_config)

    if global_vad_model is None:
        logging.info("初始化全局VAD配置...")
        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = args.silero_vad_model
        config.silero_vad.threshold = 0.5
        config.silero_vad.min_speech_duration = 0.25
        config.silero_vad.max_speech_duration = 8
        config.sample_rate = sample_rate
        config.silero_vad.window_size = 512
        global_vad_model = config
        global_vad_window_size = config.silero_vad.window_size

    if hotword_matcher is None and args.hotwords_file:
        hotword_matcher = HotwordMatcher(args.hotwords_file)

    if not replacement_rules and args.replace_rules:
        replacement_rules = load_replacement_rules(args.replace_rules)

    logging.info("全局资源初始化完成")


class ClientState:
    def __init__(self, args, client_id):
        self.client_id = client_id
        # 独立VAD实例
        self.vad = sherpa_onnx.VoiceActivityDetector(global_vad_model, buffer_size_in_seconds=85)
        # 独立状态
        self.buffer = []
        self.started = False
        self.offset = 0
        self.started_time = None
        self.last_process_time =  time.perf_counter()  # 最后处理时间
        self.total_processing_time = 0.0  # 累计处理时间
        self.process_count = 0  # 处理次数


def get_client_state(client_id: str, args=None):
    """获取或创建客户端专用状态"""
    with state_lock:
        if client_id not in client_states:
            if args is None:
                args = get_args()
            client_states[client_id] = ClientState(args, client_id)
        return client_states[client_id]


def remove_client_state(client_id: str):
    """清理客户端状态"""
    with state_lock:
        client_states.pop(client_id, None)


def update_frontend(client_id: str, text: str):
    """更新前端显示并绑定客户端ID"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 截断长文本
    display_text = text[:MAX_LOG_LENGTH] + ('...' if len(text) > MAX_LOG_LENGTH else '')
    logging.debug(f"[{client_id}] 识别结果: {display_text} [{timestamp}]")
    result_queue.put((client_id, text))


# 全局变量
result_queue = queue.Queue()


def get_args():
    """命令行参数解析器"""
    # 模型路径（写死为容器内路径）
    silero_vad_model = "model/sense-voice/silero_vad.onnx"
    tokens = "model/sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/tokens.txt"
    fire_red_asr_encoder = "model/sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/encoder.int8.onnx"
    fire_red_asr_decoder = "model/sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/decoder.int8.onnx"
    # 可选参数（可留空）
    hotwords_file = "file/hotwords.txt"
    replace_rules = "file/replace_rules.txt"
    hr_dict_dir = ""
    hr_lexicon = ""
    hr_rule_fsts = ""
    num_threads = 5  # 默认线程数

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--hotwords-file", type=str, default=hotwords_file, help="包含热词的文件路径")
    parser.add_argument("--replace-rules", type=str, default=replace_rules, help="包含替换规则的文件路径")
    parser.add_argument("--silero-vad-model", type=str, default=silero_vad_model, help="Silero VAD模型路径")
    parser.add_argument("--tokens", type=str, default=tokens, help="token文件路径")
    parser.add_argument("--fire-red-asr-encoder", default=fire_red_asr_encoder, type=str,
                        help="FireRed ASR编码器模型路径")
    parser.add_argument("--fire-red-asr-decoder", default=fire_red_asr_decoder, type=str,
                        help="FireRed ASR解码器模型路径")
    parser.add_argument("--num-threads", type=int, default=num_threads, help="推理线程数")
    parser.add_argument("--hr-dict-dir", type=str, default=hr_dict_dir, help="jieba字典目录")
    parser.add_argument("--hr-lexicon", type=str, default=hr_lexicon, help="同音词词典文件")
    parser.add_argument("--hr-rule-fsts", type=str, default=hr_rule_fsts, help="替换规则FST文件")
    return parser.parse_args()


def assert_file_exists(filename: str):
    """验证文件存在"""
    assert Path(filename).is_file(), (
        f"{filename} does not exist!\n"
        "Please refer to https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html to download it"
    )


def create_recognizer(args):
    """创建语音识别器"""
    logging.info("开始加载模型...")
    recognizer = sherpa_onnx.OfflineRecognizer.from_fire_red_asr(
        encoder=args.fire_red_asr_encoder,
        decoder=args.fire_red_asr_decoder,
        tokens=args.tokens,
        num_threads=args.num_threads,
        provider="cuda",
        debug=False,
        hr_dict_dir=args.hr_dict_dir,
        hr_rule_fsts=args.hr_rule_fsts,
        hr_lexicon=args.hr_lexicon,
    )
    logging.info("模型加载成功")
    return recognizer


def start_recognition():
    """启动语音识别服务（全局）"""
    threading.Thread(target=main_loop, daemon=True, name="RecognitionThread").start()


def stop_recognition():
    """停止语音识别服务"""
    kill_flag.set()


def main_loop():
    """全局主循环，处理所有客户端"""
    args = get_args()
    assert_file_exists(args.tokens)
    assert_file_exists(args.silero_vad_model)
    initialize_global_resources(args)

    try:
        logging.info("音频处理引擎已启动")
        while not kill_flag.is_set():
            current_time =  time.perf_counter()
            active_clients = 0

            for client_id in list(client_queues.keys()):
                try:
                    q = get_client_queue(client_id)
                    if q is None or q.empty():
                        continue

                    active_clients += 1
                    state = get_client_state(client_id, args)

                    # 获取音频数据
                    samples = q.get(timeout=0.01)
                    process_start = time.perf_counter()

                    # 添加到缓冲区
                    state.buffer = np.concatenate([state.buffer, samples])

                    # VAD处理开始时间
                    vad_start = time.perf_counter()
                    # VAD处理
                    while state.offset + global_vad_window_size < len(state.buffer):
                        state.vad.accept_waveform(state.buffer[state.offset: state.offset + global_vad_window_size])
                        if not state.started and state.vad.is_speech_detected():
                            state.started = True
                            state.started_time =  time.perf_counter()
                        state.offset += global_vad_window_size
                    vad_end =  time.perf_counter()

                    # 清理静默数据
                    if not state.started and len(state.buffer) > 10 * global_vad_window_size:
                        state.offset -= len(state.buffer) - 10 * global_vad_window_size
                        state.buffer = state.buffer[-10 * global_vad_window_size:]

                    # 语音段处理时间
                    asr_start = time.perf_counter()
                    while not state.vad.empty():
                        # 创建流
                        stream = global_recognizer.create_stream()
                        # 接受波形
                        stream.accept_waveform(sample_rate, state.vad.front.samples)
                        # 移除已处理数据
                        state.vad.pop()
                        # 解码
                        global_recognizer.decode_stream(stream)
                        # 获取结果
                        text = stream.result.text.strip()

                        # 后处理
                        if hotword_matcher and text:
                            text = hotword_matcher.match(text)
                        if text:
                            text_with_punct = global_punct.add_punctuation(text)
                            text_with_punct = apply_replacement_rules(text_with_punct, replacement_rules)
                            update_frontend(client_id, text_with_punct)
                    asr_end =  time.perf_counter()

                    # 更新状态
                    state.buffer = state.buffer[state.offset:]
                    state.offset = 0
                    if state.started and current_time - state.started_time > 8:
                        state.started = False
                        state.started_time = None

                    # 计算处理时间
                    total_time =  time.perf_counter()  - process_start
                    vad_time = vad_end - vad_start
                    asr_time = asr_end - asr_start
                    state.total_processing_time += total_time
                    state.process_count += 1

                    # 修改日志输出部分：
                    if total_time > 0:
                        vad_percent = vad_time / total_time * 100
                        asr_percent = asr_time / total_time * 100
                    else:
                        vad_percent = 0.0
                        asr_percent = 0.0

                    # 输出详细日志
                    logging.debug(f"[{client_id}] 处理统计: "
                                  f"总耗时={total_time * 1000:.2f}ms, "
                                  f"VAD耗时={vad_time * 1000:.2f}ms ({vad_percent:.1f}%), "
                                  f"ASR耗时={asr_time * 1000:.2f}ms ({asr_percent:.1f}%)")

                except queue.Empty:
                    continue
                except Exception as e:
                    logging.error(f"[{client_id}] 识别异常: {e}", exc_info=True)

            # 每5秒输出性能统计
            if current_time - main_loop.last_stat_time >= 5:
                main_loop.last_stat_time = current_time
                logging.info(f"系统统计: 活跃客户端={active_clients}, "
                             f"全局处理时间={main_loop.global_processing_time:.4f}s, "
                             f"平均帧大小={main_loop.avg_frame_size:.2f}ms")

            time.sleep(0.01)  # 避免CPU过载

    except Exception as e:
        logging.error(f"全局主循环异常: {e}", exc_info=True)
    finally:
        logging.info("全局识别服务已停止")


# 为main_loop添加静态变量
main_loop.last_stat_time =  time.perf_counter()
main_loop.global_processing_time = 0.0
main_loop.avg_frame_size = 0.0


class HotwordMatcher:
    def __init__(self, hotwords_file=None):
        self.hotwords = []
        if hotwords_file:
            self.load_hotwords(hotwords_file)

    def load_hotwords(self, hotwords_file):
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

        text_pinyin = lazy_pinyin(text, style=Style.NORMAL)
        matched = False
        new_text = list(text)
        sorted_hotwords = sorted(self.hotwords, key=lambda x: len(x[1]), reverse=True)

        i = 0
        while i < len(text_pinyin):
            replaced = False
            for word, hotword_pinyin in sorted_hotwords:
                length = len(hotword_pinyin)
                if i + length > len(text_pinyin):
                    continue
                sub_pinyin = text_pinyin[i:i + length]
                scores = [fuzz.ratio(tp, hp) for tp, hp in zip(sub_pinyin, hotword_pinyin)]
                if all(score > 90 for score in scores) and sum(scores) / len(scores) > 95:
                    new_text[i:i + length] = list(word)
                    matched = True
                    i += length
                    replaced = True
                    break
            if not replaced:
                i += 1
        return ''.join(new_text) if matched else text


def load_replacement_rules(file_path):
    """加载文本替换规则"""
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
    rules.sort(key=lambda x: len(x[0]), reverse=True)
    return rules


def apply_replacement_rules(text, rules):
    """应用文本替换规则"""
    for original, replacement in rules:
        if original in text:
            logging.debug(f"应用替换规则: {original} -> {replacement}")
            text = text.replace(original, replacement)
    return text


# 标点恢复模型
punct_model_path = "model/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx"
if not Path(punct_model_path).is_file():
    raise FileNotFoundError(f"标点模型不存在: {punct_model_path}")
punct_config = sherpa_onnx.OfflinePunctuationConfig(
    model=sherpa_onnx.OfflinePunctuationModelConfig(ct_transformer=punct_model_path),
)
global_punct = sherpa_onnx.OfflinePunctuation(punct_config)


# def main():
#     """主程序逻辑"""
#     args = get_args()
#     assert_file_exists(args.tokens)
#     assert_file_exists(args.silero_vad_model)
#     assert args.num_threads > 0, args.num_threads
#
#     # 模拟客户端处理
#     client_id = "test_client"
#     add_client_queue(client_id)
#     add_client_state(client_id, args)
#
#     # 模拟录音
#     def mock_recording():
#         while True:
#             samples = np.random.randn(512).astype(np.float32)
#             get_client_queue(client_id).put(samples)
#             time.sleep(0.032)  # 模拟实时音频流
#
#     threading.Thread(target=mock_recording, daemon=True).start()
#
#     # 启动识别
#     main_loop(client_id)
