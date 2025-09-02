# tts_converter.py

import ChatTTS
import numpy as np
import os
from queue import Empty
import concurrent.futures
from collections import deque
import time
from config_loader import config
import pickle
import cn2an
import re

NUM_WORKERS = 2 
MODEL_PATH = config['chat_tts_path']
SPEAKER_EMB_PATH = config['speaker_embedding_path']

def _clear_queue(q):
    while not q.empty():
        try: q.get_nowait()
        except Empty: break

def normalize_mixed_text(text):
    """
    在中英文、数字之间添加空格，以优化ChatTTS的处理效果。
    """
    # 正则表达式：匹配任何非中文、非字母、非数字的字符，以及字母和数字序列
    # \u4e00-\u9fa5 : 中文字符范围
    # a-zA-Z0-9 : 字母和数字
    # [^\u4e00-\u9fa5a-zA-Z0-9] : 匹配所有不是中文、字母、数字的字符 (例如标点)
    # ([a-zA-Z0-9\s'._-]+) : 匹配连续的英文、数字、空格和一些特殊字符，形成一个单词/词组
    parts = re.split(r'([a-zA-Z0-9\s\'._-]+)', text)
    
    # 过滤掉空的字符串
    parts = [p for p in parts if p]
    
    # 重新组合，确保每个部分之间有空格
    # 如果一个部分是中文，而下一个部分是英文/数字，它们之间会被一个空格隔开
    return ' '.join(parts).replace('  ', ' ') # 替换多余的空格


def convert_text_to_audio(text_queue, audio_queue, command_queue):
    print("ChatTTS 转换器正在启动...")
    try:
        chat = ChatTTS.Chat()
        chat.load(custom_path=MODEL_PATH, compile=True)
        print("ChatTTS 模型加载成功。")
    except Exception as e:
        print(f"初始化 ChatTTS 失败: {e}")
        audio_queue.put(None)
        return

    if os.path.exists(SPEAKER_EMB_PATH):
        try:
            with open(SPEAKER_EMB_PATH, 'rb') as f: spk_emb = pickle.load(f)
            print(f"已从 '{SPEAKER_EMB_PATH}' 加载保存的音色。")
        except Exception as e: spk_emb = None
    else:
        spk_emb = None
    if spk_emb is None:
        print("未找到或加载音色文件失败，正在生成随机音色...")
        spk_emb = chat.sample_random_speaker()
        with open(SPEAKER_EMB_PATH, 'wb') as f: pickle.dump(spk_emb, f)
        print(f"新音色已生成并保存到 '{SPEAKER_EMB_PATH}'。")

    params_infer_code = ChatTTS.Chat.InferCodeParams(spk_emb=spk_emb, temperature=0.6, top_P=0.7, top_K=20)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_deque = deque()
        stop_signal_received = False
        while not stop_signal_received or future_deque:
            try:
                command = command_queue.get_nowait()
                if command == "CLEAR":
                    print("[TTS Converter]: 收到CLEAR命令，清空待办任务。")
                    _clear_queue(text_queue)
                    for future in future_deque: future.cancel()
                    future_deque.clear()
            except Empty:
                pass

            if not stop_signal_received:
                try:
                    original_text = text_queue.get(timeout=0.1)
                    if original_text is None:
                        stop_signal_received = True
                    else:
                        normalized_text = normalize_mixed_text(original_text.strip())
                        text_to_speak = cn2an.transform(normalized_text, "an2cn")
                        if text_to_speak:
                            print(f"\n[音频合成任务提交]: {text_to_speak} (原始文本: {original_text.strip()})")
                            future = executor.submit(chat.infer, [text_to_speak], params_infer_code=params_infer_code)
                            future_deque.append(future)
                except Empty:
                    pass
            
            if future_deque and future_deque[0].done():
                future = future_deque.popleft()
                try:
                    wavs = future.result()
                    
                    # --- 这里是新增的诊断日志 ---
                    print(f"[TTS DEBUG]: 转换任务完成。返回结果类型: {type(wavs)}")
                    if isinstance(wavs, (list, tuple)):
                        print(f"[TTS DEBUG]: 返回结果是一个列表/元组，长度为: {len(wavs)}")
                    # ---------------------------
                    
                    if isinstance(wavs, (list, tuple)) and len(wavs) > 0:
                        audio_data = np.array(wavs[0])
                        if isinstance(audio_data, np.ndarray) and audio_data.size > 0:
                            print(f"[TTS DEBUG]: 成功提取音频数据 (大小: {audio_data.size})，准备放入播放队列。")
                            audio_queue.put(audio_data)
                        else:
                            print("[TTS DEBUG]: 警告: 返回的列表内容无效或为空数组。")
                    else:
                        print("[TTS DEBUG]: 警告: TTS模型返回的结果不是有效列表或为空。音频无法播放。")

                except (concurrent.futures.CancelledError, Exception) as e:
                    if not isinstance(e, concurrent.futures.CancelledError):
                        print(f"!!! 获取任务结果时出错: {e}")
            else:
                time.sleep(0.05)
            
            if stop_signal_received and not future_deque:
                break

    print("所有TTS任务已完成，向播放器发送结束信号。")
    audio_queue.put(None)
    print("ChatTTS 转换器已关闭。")