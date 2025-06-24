import os
import time
from datetime import datetime
import numpy as np
import websockets
import asyncio
import sounddevice as sd
import argparse

from concurrent.futures import ThreadPoolExecutor

# 音频配置参数
sample_rate = 16000  # 采样率
window_size = 512  # VAD窗口大小（与模型保持一致）

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0, help="麦克风设备ID")
parser.add_argument("--show-volume", action="store_true", help="显示实时音量")
args = parser.parse_args()

# 创建线程池执行器（CPU核心数的一半）
executor = ThreadPoolExecutor(max_workers=os.cpu_count() // 2)


async def receive_results(websocket):
    """接收识别结果并测量端到端延迟"""
    try:
        async for message in websocket:
            # 获取当前时间戳
            receive_time = time.perf_counter()

            # 检查是否有发送时间戳记录
            if hasattr(receive_results, 'last_send_time'):
                # 计算端到端延迟（毫秒）
                latency = (receive_time - receive_results.last_send_time) * 1000

                # 显示当前时间
                current_time = datetime.now().strftime("%H:%M:%S")

                # 打印结果与延迟
                print(f"[时间] {current_time}")
                print(f"\r[识别结果] {message}  ")
                print(f"[延迟] 端到端耗时: {latency:.2f}ms")
            else:
                print("未找到发送时间戳记录")

    except websockets.exceptions.ConnectionClosed:
        print("WebSocket连接已关闭")


async def send_audio(websocket, device_id):
    """发送音频数据并监控性能指标"""
    loop = asyncio.get_event_loop()

    # 打开麦克风输入流
    with sd.InputStream(device=device_id, channels=1, dtype="float32", samplerate=sample_rate) as s:
        print(f"麦克风已打开 (设备 {device_id})，开始发送音频数据...")

        # 初始化缓冲区与性能统计变量
        buffer = []
        volume_counter = 0
        total_start_time = time.perf_counter()
        frame_count = 0
        total_audio_read_time = 0
        total_volume_time = 0
        total_send_time = 0
        total_loop_time = 0

        while True:
            # ★★★ 异步化音频采集（关键优化）
            audio_read_start = time.perf_counter()
            samples, _ = await loop.run_in_executor(executor, s.read, window_size)
            audio_read_end = time.perf_counter()

            # ★★★ 实时音量检测
            volume_start = time.perf_counter()
            volume_counter += 1  # 移出条件判断
            if volume_counter % 5 == 0:
                rms = np.sqrt(np.mean(samples ** 2))  # RMS音量值
                print(f"\r[音量] {rms:.2f}         ", end="")
            volume_end = time.perf_counter()

            # ★★★ 批量发送优化
            send_start = time.perf_counter()
            buffer.append(samples)

            # 达到批量阈值时发送
            if len(buffer) >= 4:  # 批量发送4个数据块
                combined = np.concatenate(buffer)

                # ★★★ 记录发送时间戳
                receive_results.last_send_time = time.perf_counter()

                await websocket.send(combined.tobytes())
                buffer.clear()
            send_end = time.perf_counter()

            # 更新性能统计
            frame_count += 1
            total_audio_read_time += (audio_read_end - audio_read_start)
            total_volume_time += (volume_end - volume_start)
            total_send_time += (send_end - send_start)
            total_loop_time += (time.perf_counter() - audio_read_start)

            # 性能监控（调试用）
            if hasattr(send_audio, 'start_time'):
                send_audio.samples_processed += window_size
                if time.perf_counter() - send_audio.start_time >= 1:
                    current_loop_time = time.perf_counter() - send_audio.start_time

                    # 计算各阶段占比
                    read_percent = (total_audio_read_time / current_loop_time) * 100
                    volume_percent = (total_volume_time / current_loop_time) * 100
                    send_percent = (total_send_time / current_loop_time) * 100

                    # 打印性能报告
                    print(f"\r[性能] 采样率: {send_audio.samples_processed}Hz | "
                          f"读取: {read_percent:.1f}% | "
                          f"音量: {volume_percent:.1f}% | "
                          f"发送: {send_percent:.1f}% | "
                          f"延迟: {current_loop_time * 1000:.2f}ms")

                    # 重置统计
                    send_audio.samples_processed = 0
                    send_audio.start_time = time.perf_counter()
                    total_audio_read_time = 0
                    total_volume_time = 0
                    total_send_time = 0
                    total_loop_time = 0
            else:
                # 初始化性能统计
                send_audio.start_time = time.perf_counter()
                send_audio.samples_processed = 0


async def test_audio_stream():
    """主测试任务"""
    try:
        async with websockets.connect(
                "ws://localhost:9090/ws/audio",
                # "ws://192.168.70.161:9090/ws/audio",
                ping_interval=20,
                ping_timeout=20,
                # timeout=30  # 添加连接超时
        ) as websocket:
            print("已连接到 WebSocket 服务器")

            # 启动双向通信任务
            tasks = [
                asyncio.create_task(send_audio(websocket, args.device)),
                asyncio.create_task(receive_results(websocket))
            ]

            await asyncio.gather(*tasks)  # 等待所有任务完成

    except asyncio.TimeoutError:
        print("连接超时！请检查网络连接")
    except Exception as e:
        print(f"发生异常: {e}")


# 修改后的main函数
if __name__ == "__main__":
    # 创建独立事件循环
    loop = asyncio.new_event_loop()
    # 创建线程池并绑定事件循环
    executor = ThreadPoolExecutor(max_workers=os.cpu_count() // 2)
    loop.set_default_executor(executor)

    try:
        # 运行主任务
        loop.run_until_complete(test_audio_stream())

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        print("等待现有任务完成...")
        # 获取活跃任务（在事件循环关闭前）
        tasks = [t for t in asyncio.all_tasks(loop) if not t.done()]

        # 取消所有任务
        for task in tasks:
            task.cancel()

        # 等待任务安全终止（设置超时）
        try:
            loop.run_until_complete(
                asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=2)
            )
        except asyncio.TimeoutError:
            print("任务等待超时，强制终止")

        # 关闭线程池（先于事件循环关闭）
        executor.shutdown(wait=True)

        # 关闭事件循环
        if not loop.is_closed():
            loop.stop()
            loop.close()

        print("资源已安全释放")

