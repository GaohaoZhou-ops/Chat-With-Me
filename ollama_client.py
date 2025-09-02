import ollama
import openai
import re

FIRST_CHUNK_MIN_LENGTH = 18
MAX_CHARS_PER_CHUNK = 50

def _process_and_queue_text_chunk(text_chunk, text_queue, ui_queue):
    text_chunk = text_chunk.strip()
    if not text_chunk:
        return

    def queue_chunk(chunk_to_queue):
        text_queue.put(chunk_to_queue)
        if ui_queue:
            ui_queue.put(chunk_to_queue)

    if len(text_chunk) <= MAX_CHARS_PER_CHUNK:
        queue_chunk(text_chunk)
        return

    print(f"\n[文本切分]: 检测到长句 (长度 {len(text_chunk)} > {MAX_CHARS_PER_CHUNK})...")
    
    parts = re.split(r'([，；,;])', text_chunk)
    current_chunk = ""
    for i in range(0, len(parts), 2):
        part = parts[i]
        delimiter = parts[i+1] if i + 1 < len(parts) else ""
        full_part = part + delimiter

        if len(current_chunk) + len(full_part) > MAX_CHARS_PER_CHUNK and current_chunk:
            queue_chunk(current_chunk.strip())
            current_chunk = full_part
        else:
            current_chunk += full_part
    
    if current_chunk:
        queue_chunk(current_chunk.strip())


def stream_ollama_response(input_queue, text_queue, local_model_config, system_prompt, ui_queue=None):
    model_name = local_model_config['name']
    print(f"Ollama 客户端已启动，使用模型: {model_name}")

    while True:
        prompt = input_queue.get()
        if prompt is None:
            text_queue.put(None)
            if ui_queue: ui_queue.put(None)
            break
        
        print(f"\n[用户]: {prompt}")
        print("[AI]: ", end="", flush=True)
        
        full_sentence = ""
        is_first_chunk = True

        try:
            stream = ollama.chat(model=model_name, messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': prompt}], stream=True)
            for chunk in stream:
                content = chunk['message']['content']
                if content:
                    print(content, end="", flush=True)
                    full_sentence += content

                    if is_first_chunk and len(full_sentence) >= FIRST_CHUNK_MIN_LENGTH:
                        print("\n[快速响应]: 检测到首个文本块，优先合成...")
                        _process_and_queue_text_chunk(full_sentence, text_queue, ui_queue)
                        full_sentence = ""
                        is_first_chunk = False
                        continue

                    if not is_first_chunk:
                        sentences = re.split(r'(?<=[。！？\!\?])\s*', full_sentence)
                        if len(sentences) > 1:
                            for sentence in sentences[:-1]:
                                _process_and_queue_text_chunk(sentence, text_queue, ui_queue)
                            full_sentence = sentences[-1]
        except Exception as e:
            print(f"\n调用 Ollama 时出错: {e}")
        
        if full_sentence.strip():
            _process_and_queue_text_chunk(full_sentence, text_queue, ui_queue)
        
        if ui_queue:
            ui_queue.put(None)
        print()

def stream_openai_response(input_queue, text_queue, online_model_config, system_prompt, ui_queue=None):
    model_name = online_model_config['name']
    api_key = online_model_config['api_key']
    base_url = online_model_config['base_url']
    print(f"OpenAI 客户端已启动，使用模型: {model_name}, API 地址: {base_url}")

    try:
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
    except Exception as e:
        print(f"初始化 OpenAI 客户端失败: {e}")
        text_queue.put(None)
        if ui_queue: ui_queue.put(None)
        return

    while True:
        prompt = input_queue.get()
        if prompt is None:
            text_queue.put(None)
            if ui_queue: ui_queue.put(None)
            break
        
        print(f"\n[用户]: {prompt}")
        print("[AI]: ", end="", flush=True)

        full_sentence = ""
        is_first_chunk = True

        try:
            stream = client.chat.completions.create(model=model_name, messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': prompt}], stream=True, temperature=0.7)
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    print(content, end="", flush=True)
                    full_sentence += content
                    
                    if is_first_chunk and len(full_sentence) >= FIRST_CHUNK_MIN_LENGTH:
                        print("\n[快速响应]: 检测到首个文本块，优先合成...")
                        _process_and_queue_text_chunk(full_sentence, text_queue, ui_queue)
                        full_sentence = ""
                        is_first_chunk = False
                        continue

                    if not is_first_chunk:
                        sentences = re.split(r'(?<=[。！？\!\?])\s*', full_sentence)
                        if len(sentences) > 1:
                            for sentence in sentences[:-1]:
                               _process_and_queue_text_chunk(sentence, text_queue, ui_queue)
                            full_sentence = sentences[-1]
        except Exception as e:
            print(f"\n调用 OpenAI API 时发生未知错误: {e}")
        
        if full_sentence.strip():
            _process_and_queue_text_chunk(full_sentence, text_queue, ui_queue)
        
        if ui_queue:
            ui_queue.put(None)
        print()