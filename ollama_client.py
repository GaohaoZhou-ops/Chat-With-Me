# ollama_client.py

import ollama
import openai
import re

# 定义单个文本块的最大字符数
MAX_CHARS_PER_CHUNK = 50

def _process_and_queue_text_chunk(text_chunk, text_queue):
    """
    处理并发送文本块到队列。
    如果文本块太长，会根据逗号等标点进行二次切分。
    """
    text_chunk = text_chunk.strip()
    if not text_chunk:
        return

    if len(text_chunk) <= MAX_CHARS_PER_CHUNK:
        text_queue.put(text_chunk)
        return

    print(f"\n[文本切分]: 检测到长句 (长度 {len(text_chunk)} > {MAX_CHARS_PER_CHUNK})，尝试按逗号/分号切分...")
    
    parts = re.split(r'([，；,;])', text_chunk)
    
    current_chunk = ""
    for i in range(0, len(parts), 2):
        part = parts[i]
        delimiter = parts[i+1] if i + 1 < len(parts) else ""
        full_part = part + delimiter

        if len(current_chunk) + len(full_part) > MAX_CHARS_PER_CHUNK and current_chunk:
            text_queue.put(current_chunk.strip())
            current_chunk = full_part
        else:
            current_chunk += full_part
    
    if current_chunk:
        text_queue.put(current_chunk.strip())

def stream_ollama_response(input_queue, text_queue, local_model_config, system_prompt):
    """
    从配置中获取模型名和系统提示词。
    """
    model_name = local_model_config['name']
    print(f"Ollama 客户端已启动，使用模型: {model_name}")
    print(f"使用的系统提示词: \"{system_prompt.strip()}\"")

    while True:
        prompt = input_queue.get()
        if prompt is None:
            text_queue.put(None)
            break

        print(f"\n[用户]: {prompt}")
        # print("[AI]: ", end="", flush=True)
        
        full_sentence = ""
        try:
            stream = ollama.chat(
                model=model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt}
                ],
                stream=True,
            )
            
            for chunk in stream:
                content = chunk['message']['content']
                if content:
                    print(content, end="", flush=True)
                    full_sentence += content
                    sentences = re.split(r'(?<=[。！？\!\?])\s*', full_sentence)
                    
                    if len(sentences) > 1:
                        for sentence in sentences[:-1]:
                            _process_and_queue_text_chunk(sentence, text_queue)
                        full_sentence = sentences[-1]

        except Exception as e:
            print(f"\n调用 Ollama 时出错: {e}")
            continue
        
        if full_sentence.strip():
            _process_and_queue_text_chunk(full_sentence, text_queue)
        print()

# --- 注意这个函数的定义 ---
def stream_openai_response(input_queue, text_queue, online_model_config, system_prompt):
    """
    从配置中获取模型名、API Key、Base URL和系统提示词。
    """
    # 从 online_model_config 字典中解包出所需的值
    model_name = online_model_config['name']
    api_key = online_model_config['api_key']
    base_url = online_model_config['base_url']

    print(f"OpenAI 客户端已启动，使用模型: {model_name}, API 地址: {base_url}")
    print(f"使用的系统提示词: \"{system_prompt.strip()}\"")
    
    try:
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
    except Exception as e:
        print(f"初始化 OpenAI 客户端失败: {e}")
        text_queue.put(None)
        return

    while True:
        prompt = input_queue.get()
        if prompt is None:
            text_queue.put(None)
            break

        print(f"\n[用户]: {prompt}")
        print("[AI]: ", end="", flush=True)

        full_sentence = ""
        try:
            stream = client.chat.completions.create(
                model=model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt}
                ],
                stream=True,
                temperature=0.7,
            )

            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    print(content, end="", flush=True)
                    full_sentence += content
                    sentences = re.split(r'(?<=[。！？\!\?])\s*', full_sentence)
                    
                    if len(sentences) > 1:
                        for sentence in sentences[:-1]:
                           _process_and_queue_text_chunk(sentence, text_queue)
                        full_sentence = sentences[-1]

        except Exception as e:
            print(f"\n调用 OpenAI API 时发生未知错误: {e}")
            continue

        if full_sentence.strip():
            _process_and_queue_text_chunk(full_sentence, text_queue)
        print()