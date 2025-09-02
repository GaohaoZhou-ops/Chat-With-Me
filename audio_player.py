import sounddevice as sd
import queue

def clear_queue(q):
    """
    安全地清空一个队列。
    """
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break

def play_audio_data(audio_queue, command_queue):
    """
    健壮的音频播放逻辑，能处理所有状态并响应命令。
    """
    print("音频播放器进程已启动，等待音频或命令...")
    while True:
        try:
            # 1. 首先以带超时的方式等待音频数据
            # 超时机制确保我们能周期性地检查命令队列
            audio_data = audio_queue.get(timeout=0.1)

            if audio_data is None:
                print("音频播放器收到结束信号，即将关闭。")
                break
            
            # 2. 收到音频数据，开始播放
            sd.play(audio_data, samplerate=24000)
            print("[Player]: 开始播放音频...")

            # 3. 进入内部循环，监控播放状态和命令
            # 只有在 sd.play() 调用后，这个循环才是安全的
            while sd.get_stream().active:
                try:
                    # 检查命令队列
                    command = command_queue.get_nowait()
                    if command == "CLEAR":
                        print("[Player]: 在播放期间收到CLEAR命令，停止并清空队列。")
                        sd.stop()
                        clear_queue(audio_queue)
                        break  # 跳出内部循环
                except queue.Empty:
                    # 没有命令，短暂休眠
                    sd.sleep(50)
            
            print("[Player]: 本次播放结束。")

        except queue.Empty:
            # 音频队列为空，这是正常情况。我们在这里检查命令队列。
            try:
                command = command_queue.get_nowait()
                if command == "CLEAR":
                    print("[Player]: 在空闲时收到CLEAR命令，清空队列。")
                    clear_queue(audio_queue)
            except queue.Empty:
                # 没有音频，也没有命令，继续等待
                pass
            continue

        except Exception as e:
            print(f"播放音频时发生未知错误: {e}")
            break
            
    print("音频播放器进程已关闭。")