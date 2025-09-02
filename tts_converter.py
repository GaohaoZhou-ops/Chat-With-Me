# tts_converter.py

import ChatTTS
import numpy as np
from queue import Empty

def convert_text_to_audio(text_queue, audio_queue):
    """
    从文本队列获取句子，合成为音频，并将音频数据放入音频队列。
    不再负责播放。
    """
    print("ChatTTS 转换器已启动，等待文本...")
    
    try:
        chat = ChatTTS.Chat()
        chat.load(custom_path="/Users/gaohao/Desktop/ChatTTS", compile=False) 
        print("ChatTTS 模型加载成功。")
    except Exception as e:
        print(f"初始化 ChatTTS 失败: {e}")
        audio_queue.put(None) # 如果初始化失败，通知下游进程退出
        return

    rand_spk = chat.sample_random_speaker()
    print("生成随机音色...")

    params_infer_code = ChatTTS.Chat.InferCodeParams(
        spk_emb=rand_spk,
        temperature=0.6,
        top_P=0.7,
        top_K=20
    )

    sentence_buffer = []

    while True:
        try:
            # 尝试从队列中获取内容，最长等待1.0秒
            text = text_queue.get(timeout=1.0)

            if text is None: # 收到上游的结束信号
                # 在退出前，处理缓冲区中最后剩余的任何内容
                if sentence_buffer:
                    text_to_speak = " ".join(sentence_buffer)
                    print(f"\n[音频合成中 - 结尾]: {text_to_speak}")
                    wavs = chat.infer([text_to_speak], params_infer_code=params_infer_code, use_decoder=True)
                    audio_data = np.array(wavs[0])
                    audio_queue.put(audio_data) # 放入音频队列
                    sentence_buffer.clear()
                
                audio_queue.put(None) # 向下游的播放器进程发送结束信号
                break # 退出循环

            sentence_buffer.append(text.strip())
            
            # 将缓冲区大小调整为2，以便更快地获得音频反馈
            if len(sentence_buffer) >= 2:
                text_to_speak = " ".join(sentence_buffer)
                print(f"\n[音频合成中]: {text_to_speak}")
                wavs = chat.infer([text_to_speak], params_infer_code=params_infer_code, use_decoder=True)
                audio_data = np.array(wavs[0])
                audio_queue.put(audio_data) # 放入音频队列
                sentence_buffer.clear()

        except Empty:
            # 如果超时且缓冲区有内容，则处理它们
            if sentence_buffer:
                text_to_speak = " ".join(sentence_buffer)
                print(f"\n[音频合成中 - 超时]: {text_to_speak}")
                wavs = chat.infer([text_to_speak], params_infer_code=params_infer_code, use_decoder=True)
                audio_data = np.array(wavs[0])
                audio_queue.put(audio_data) # 放入音频队列
                sentence_buffer.clear()

    print("ChatTTS 转换器已关闭。")