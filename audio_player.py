# audio_player.py

import sounddevice as sd

def play_audio_data(audio_queue):
    """
    从音频队列中获取音频数据并播放。
    这是一个独立的消费者进程。
    """
    print("音频播放器进程已启动，等待音频数据...")
    while True:
        try:
            audio_data = audio_queue.get()
            if audio_data is None:  # 收到结束信号
                print("音频播放器收到结束信号，即将关闭。")
                break
            
            # 播放获取到的音频数据
            sd.play(audio_data, samplerate=24000)
            sd.wait()
        except Exception as e:
            print(f"播放音频时出错: {e}")
            break
            
    print("音频播放器进程已关闭。")