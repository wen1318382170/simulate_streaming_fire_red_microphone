import logging
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import asyncio
from threading import Thread
from typing import Dict, Deque
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from starlette.middleware.cors import CORSMiddleware
from simulate_streaming_fire_red_microphone import (
    start_recognition, stop_recognition,
    result_queue, kill_flag, sample_rate,
    add_client_queue, remove_client_queue, get_client_queue
)

# 使用弱引用避免内存泄漏
import weakref

connected_clients = weakref.WeakValueDictionary()


# 异步批量推送管理器
class BatchPushManager:
    def __init__(self):
        self.batches = {}  # client_id -> messages
        self.lock = threading.Lock()
        self.max_batch_size = 10  # 最大批量消息数
        self.flush_interval = 0.05  # 默认刷新间隔

    async def add_message(self, client_id, message):
        with self.lock:
            if client_id not in self.batches:
                self.batches[client_id] = []
            self.batches[client_id].append(message)

            # 达到最大批量立即刷新
            if len(self.batches[client_id]) >= self.max_batch_size:
                await self.flush(client_id)

    async def flush(self, client_id=None):
        with self.lock:
            if client_id:
                targets = {client_id: self.batches.pop(client_id, [])}
            else:
                targets = self.batches.copy()
                self.batches.clear()

        for cid, messages in targets.items():
            if cid in connected_clients:
                try:
                    start_time = time.time()
                    await connected_clients[cid].send_text("\n".join(messages))
                    elapsed = time.time() - start_time
                    logging.debug(f"[{cid}] 结果推送耗时: {elapsed:.4f}s, 消息数: {len(messages)}")
                except WebSocketDisconnect:
                    pass


batch_pusher = BatchPushManager()


# 性能指标监控
class PerformanceMetrics:
    def __init__(self):
        self.messages_sent = 0
        self.bytes_sent = 0
        self.clients = 0
        self.start_time = time.time()
        self.last_flush_time = time.time()


metrics = PerformanceMetrics()

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

# 客户端状态管理
client_states = {}
clients_lock = threading.Lock()


# 修改lifespan函数
@asynccontextmanager
async def lifespan(app: FastAPI):
    """生命周期管理：启动后台任务"""
    global recognition_thread, background_task_thread, background_loop
    logging.info("启动全局识别服务...")
    kill_flag.clear()

    # 创建独立事件循环
    background_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(background_loop)

    # 启动识别线程
    recognition_thread = Thread(target=start_recognition, daemon=True, name="RecognitionThread")
    recognition_thread.start()

    # 启动后台任务线程
    background_task_thread = Thread(
        target=lambda: background_loop.run_until_complete(start_background_tasks()),
        daemon=True,
        name="BackgroundTaskThread"
    )
    background_task_thread.start()

    yield

    # 清理逻辑
    logging.info("开始停止全局识别服务...")
    kill_flag.set()

    # 主动关闭WebSocket连接
    for client_id in list(connected_clients.keys()):
        try:
            websocket = connected_clients[client_id]
            asyncio.run_coroutine_threadsafe(websocket.close(), background_loop)
        except Exception as e:
            logging.warning(f"[{client_id}] 主动关闭失败: {e}")

    # 请求线程停止
    if background_task_thread and background_task_thread.is_alive():
        logging.info("请求后台任务线程停止...")
        background_loop.call_soon_threadsafe(background_loop.stop)

        # 等待线程结束（带超时）
        background_task_thread.join(timeout=3)
        if background_task_thread.is_alive():
            logging.warning("后台任务线程未响应停止请求")

    # 强制清理资源
    for thread, name in [(recognition_thread, "识别"), (background_task_thread, "后台任务")]:
        if thread and thread.is_alive():
            logging.info(f"等待{name}线程结束...")
            thread.join(timeout=2)
            if thread.is_alive():
                logging.warning(f"{name}线程未在超时时间内退出")

    # 显式关闭事件循环
    if background_loop and not background_loop.is_closed():
        logging.info("关闭后台事件循环...")
        background_loop.stop()
        background_loop.close()

    logging.info("全局服务已停止")


# 创建应用
app = FastAPI(lifespan=lifespan)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],
    allow_credentials=True,
    allow_origin_regex=".*",
)


# WebSocket连接处理
@app.websocket("/ws/audio")
async def audio_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = str(id(websocket))
    logging.info(f"[{client_id}] 新客户端连接")

    # 记录连接时间
    connect_time = time.time()

    add_client_queue(client_id)

    # 添加到连接池
    connected_clients[client_id] = websocket
    metrics.clients += 1

    try:
        handshake_start = time.time()
        await websocket.send_text(f"client_ready|{client_id}")
        handshake_time = time.time() - handshake_start
        logging.debug(f"[{client_id}] 握手耗时: {handshake_time:.4f}s")

        logging.info(f"[{client_id}] 已发送握手消息")

        while True:
            try:
                receive_start = time.time()
                audio_data = await websocket.receive_bytes()
                receive_time = time.time() - receive_start

                samples = np.frombuffer(audio_data, dtype=np.float32, count=-1)
                put_start = time.time()
                get_client_queue(client_id).put_nowait(samples)
                put_time = time.time() - put_start

                metrics.bytes_sent += len(audio_data)

                logging.debug(f"[{client_id}] 数据接收统计: "
                              f"接收耗时={receive_time:.4f}s, "
                              f"入队耗时={put_time:.4f}s, "
                              f"数据大小={len(audio_data)}B")

            except WebSocketDisconnect:
                break
    finally:
        # 异步清理
        disconnect_time = time.time()
        duration = disconnect_time - connect_time
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: [
                connected_clients.pop(client_id, None),
                remove_client_queue(client_id),
                metrics.__setattr__('clients', metrics.clients - 1)
            ]
        )
        logging.info(f"[{client_id}] 客户端断开连接, 持续时间={duration:.2f}s")


# 广播结果
async def broadcast_result(text: str):
    try:
        client_id, result = text.split("|", 1)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] {result}"
        metrics.messages_sent += 1

        # 添加到批量推送队列
        batch_start = time.time()
        await batch_pusher.add_message(client_id, formatted)
        batch_time = time.time() - batch_start

        logging.debug(f"[{client_id}] 结果广播统计: "
                      f"批处理耗时={batch_time:.4f}s, "
                      f"总消息数={metrics.messages_sent}")

    except ValueError:
        logging.warning(f"[推送失败] 数据格式错误: {text}")


# 后台任务处理
def start_background_tasks():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # 使用线程池处理阻塞操作
    executor = ThreadPoolExecutor(max_workers=2)

    async def process_queue():
        last_flush = time.time()
        while not kill_flag.is_set():
            try:
                if not result_queue.empty():
                    client_id, text = result_queue.get_nowait()
                    process_start = time.time()
                    await loop.run_in_executor(executor,
                                               lambda: asyncio.run(broadcast_result(f"{client_id}|{text}")))
                    total_time = time.time() - process_start

                    logging.debug(f"[{client_id}] 队列处理耗时: {total_time:.4f}s")

                # 动态调整间隔
                queue_size = result_queue.qsize()
                if queue_size > 100:
                    delay = 0.005  # 高负载时缩短间隔
                elif queue_size < 10:
                    delay = 0.1  # 低负载时延长间隔
                else:
                    delay = 0.02

                # 定期刷新剩余消息
                if time.time() - last_flush > 0.1:
                    flush_start = time.time()
                    await batch_pusher.flush()
                    flush_time = time.time() - flush_start
                    last_flush = time.time()

                    # logging.debug(f"批量刷新耗时: {flush_time:.4f}s")

                await asyncio.sleep(delay)
            except Exception as e:
                logging.error(f"[后台任务] 异常: {e}", exc_info=True)

    async def performance_monitor():
        while not kill_flag.is_set():
            await asyncio.sleep(5)
            duration = time.time() - metrics.start_time
            tps = metrics.messages_sent / duration if duration > 0 else 0
            logging.info(f"性能统计: 客户端={metrics.clients}, "
                         f"总消息={metrics.messages_sent}, "
                         f"吞吐量={tps:.1f}msg/s, "
                         f"总流量={metrics.bytes_sent / 1024:.1f}KB")

    try:
        loop.run_until_complete(
            asyncio.gather(
                process_queue(),
                performance_monitor()
            )
        )
    finally:
        loop.close()
        executor.shutdown(wait=False)


# 启动服务
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9090)
