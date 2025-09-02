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

def convert_year_in_text(text):
    """
    更智能地将文本中的四位数字年份转换为逐字朗读的中文格式。
    例如: "1920年" -> "一九二零年", "我出生于1995" -> "我出生于一九九五"
    """
    digit_map = {'0': '零', '1': '一', '2': '二', '3': '三', '4': '四', '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'}

    def replace_year(match):
        year_digits = match.group(1)
        # 将每个数字转换为对应的中文字符
        chinese_year = ''.join(digit_map[digit] for digit in year_digits)
        
        # 检查原始匹配的末尾是否是 "年"，如果是，则在转换后的结果中也加上 "年"
        if match.group(0).endswith('年'):
            return f"{chinese_year}年"
        else:
            return chinese_year

    # 更新正则表达式：
    # \b(\d{4})\b  -> 匹配独立的四位数字 (例如 "1995")
    # (\d{4})年   -> 匹配后面跟着 "年" 的四位数字 (例如 "1995年")
    # 使用 | (或) 将两种情况合并
    # 使用非捕获组 (?:...) 来处理 "年" 的可选情况
    return re.sub(r'\b(\d{4})(?:年)?\b', replace_year, text)

def normalize_mixed_text(text):
    """
    在中英文、数字之间添加空格，并特殊处理大写缩写词，以优化ChatTTS的处理效果。
    例如："OLED电视" -> "O L E D 电视"
    """
    # 使用正则表达式切分文本，保留英文、数字和特定符号的组合
    parts = re.split(r'([a-zA-Z0-9\s\'._-]+)', text)
    
    processed_parts = []
    for part in parts:
        if not part:
            continue
        # 条件：完全由2个或以上大写英文字母组成
        if re.fullmatch(r'[A-Z]{2,}', part.strip()):
            # 是缩写词，展开为 "L E D" 的形式
            processed_parts.append(" ".join(part.strip()))
        else:
            processed_parts.append(part)
    # 重新组合，并清理多余的空格
    return ' '.join(processed_parts).replace('  ', ' ')

def _clear_queue(q):
    while not q.empty():
        try: q.get_nowait()
        except Empty: break

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
                        # 1. 标准化文本，在中英文之间添加空格
                        normalized_text = normalize_mixed_text(original_text.strip())
                        
                        # 2. 优先处理文本中的年份
                        text_with_years_converted = convert_year_in_text(normalized_text)
                        
                        # 3. 对处理完年份的文本进行其余的数字转换
                        text_to_speak = cn2an.transform(text_with_years_converted, "an2cn")

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
                    
                    print(f"[TTS DEBUG]: 转换任务完成。返回结果类型: {type(wavs)}")
                    if isinstance(wavs, (list, tuple)):
                        print(f"[TTS DEBUG]: 返回结果是一个列表/元组，长度为: {len(wavs)}")
                    
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