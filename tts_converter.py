import ChatTTS
import numpy as np
import os
from queue import Empty
import concurrent.futures
from collections import deque
import time
from config_loader import config
import pickle

NUM_WORKERS = 2 
MODEL_PATH = config['chat_tts_path']
SPEAKER_EMB_PATH = config['speaker_embedding_path']

def convert_chunk(chat_instance, text_chunk, infer_params):
    """
    工作线程实际执行的转换任务。
    """
    try:
        wavs = chat_instance.infer(
            [text_chunk], 
            params_infer_code=infer_params, 
            use_decoder=True
        )
        return np.array(wavs[0])
    except Exception as e:
        print(f"!!! 在工作线程中进行TTS转换时出错: {e}")
        return None

def convert_text_to_audio(text_queue, audio_queue):
    """
    使用线程池并行转换文本，并使用 pickle 正确地保存和加载音色。
    """
    print("ChatTTS 转换器正在启动...")
    
    try:
        chat = ChatTTS.Chat()
        chat.load(custom_path=MODEL_PATH, compile=False)
        print("ChatTTS 模型加载成功。")
    except Exception as e:
        print(f"初始化 ChatTTS 失败: {e}")
        audio_queue.put(None)
        return

    # 使用 pickle 加载或生成音色
    if os.path.exists(SPEAKER_EMB_PATH):
        try:
            with open(SPEAKER_EMB_PATH, 'rb') as f:
                spk_emb = pickle.load(f)
            print(f"已从 '{SPEAKER_EMB_PATH}' 加载保存的音色。")
        except Exception as e:
            print(f"加载音色文件失败: {e}, 将重新生成。")
            spk_emb = None
    else:
        spk_emb = None

    if spk_emb is None:
        print("未找到或加载音色文件失败，正在生成随机音色...")
        spk_emb = chat.sample_random_speaker()
        with open(SPEAKER_EMB_PATH, 'wb') as f:
            pickle.dump(spk_emb, f)
        print(f"新音色已生成并保存到 '{SPEAKER_EMB_PATH}'。")

    params_infer_code = ChatTTS.Chat.InferCodeParams(
        spk_emb=spk_emb,
        temperature=0.6,
        top_P=0.7,
        top_K=20
    )

    sentence_buffer = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_deque = deque()
        stop_signal_received = False

        while not stop_signal_received or future_deque:
            if not stop_signal_received and len(future_deque) < NUM_WORKERS * 2:
                try:
                    text = text_queue.get(timeout=0.2)
                    if text is None:
                        stop_signal_received = True
                        if sentence_buffer:
                            text_to_speak = " ".join(sentence_buffer)
                            print(f"\n[音频合成任务提交 - 结尾]: {text_to_speak}")
                            future = executor.submit(convert_chunk, chat, text_to_speak, params_infer_code)
                            future_deque.append(future)
                            sentence_buffer.clear()
                    else:
                        sentence_buffer.append(text.strip())
                        if len(sentence_buffer) >= 2:
                            text_to_speak = " ".join(sentence_buffer)
                            print(f"\n[音频合成任务提交]: {text_to_speak}")
                            future = executor.submit(convert_chunk, chat, text_to_speak, params_infer_code)
                            future_deque.append(future)
                            sentence_buffer.clear()
                except Empty:
                    if sentence_buffer:
                        text_to_speak = " ".join(sentence_buffer)
                        print(f"\n[音频合成任务提交]: {text_to_speak}")
                        future = executor.submit(convert_chunk, chat, text_to_speak, params_infer_code)
                        future_deque.append(future)
                        sentence_buffer.clear()

            if future_deque and future_deque[0].done():
                future = future_deque.popleft()
                try:
                    audio_data = future.result()
                    if audio_data is not None and audio_data.size > 0:
                        audio_queue.put(audio_data)
                except Exception as e:
                    print(f"!!! 获取任务结果时出错: {e}")
            else:
                time.sleep(0.05)
            
            if stop_signal_received and not future_deque:
                break

    print("所有TTS任务已完成，向播放器发送结束信号。")
    audio_queue.put(None)
    print("ChatTTS 转换器已关闭。")