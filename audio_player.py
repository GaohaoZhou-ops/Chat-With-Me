# audio_player.py

import sounddevice as sd

def play_audio_data(audio_queue):
    """
    从音频队列中获取音频数据并播放。
    """
    print("音频播放器进程已启动，等待音频数据...")
    while True:
        try:
            audio_data = audio_queue.get()
            print(f"[DEBUG-PLAYER] Got item from audio_queue. Is it None? {audio_data is None}") # <--- 添加这行
            if audio_data is None:
                print("音频播放器收到结束信号，即将关闭。")
                break
            
            print("[DEBUG-PLAYER] Playing audio...") # <--- 添加这行
            sd.play(audio_data, samplerate=24000)
            sd.wait()
            print("[DEBUG-PLAYER] Finished playing audio.") # <--- 添加这行
        except Exception as e:
            print(f"播放音频时出错: {e}")
            break
            
    print("音频播放器进程已关闭。")