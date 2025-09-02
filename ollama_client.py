# ollama_client.py

import ollama
import openai
import re

SYSTEM_PROMPT = """
你是一个乐于助人的人工智能助手。你的回答应该总是简洁、清晰，并尽可能提供帮助。
你的回答必须严格遵守以下规范：

1. 由于你的回答将在本地通过TTS转换为语音，因此回答中不允许有任何表情、代码、json、markdown等格式的内容
2. 如果你返回的是中文并切包含了数字，则需要将其转换成汉字形式以方便tts转换，但需要注意如1990年、5个这种转换形式
"""

# --------------------------------------------------------

def stream_ollama_response(input_queue, text_queue, model_name='qwen2:7b-instruct-q4_0'):
    """
    从输入队列获取问题，调用 Ollama 进行流式推理，
    并将生成的文本块放入文本队列。
    """
    print(f"Ollama 客户端已启动，使用模型: {model_name}")
    print(f"使用的系统提示词: \"{SYSTEM_PROMPT}\"")

    while True:
        prompt = input_queue.get()
        if prompt is None:
            text_queue.put(None)
            break

        print(f"\n[用户]: {prompt}")
        print("[AI]: ", end="", flush=True)
        
        full_sentence = ""
        try:
            print(f"\n[诊断信息]: 正在尝试连接本地 Ollama 模型: {model_name}...")

            stream = ollama.chat(
                model=model_name,
                messages=[
                    # 使用共享的系统提示词
                    {'role': 'system', 'content': SYSTEM_PROMPT},
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
                            if sentence.strip():
                                text_queue.put(sentence.strip())
                        full_sentence = sentences[-1]

        except Exception as e:
            print(f"\n调用 Ollama 时出错: {e}")
            continue
        
        if full_sentence.strip():
            text_queue.put(full_sentence.strip())
        print()


def stream_openai_response(input_queue, text_queue, model_name, api_key, base_url):
    print(f"OpenAI 客户端已启动，使用模型: {model_name}, API 地址: {base_url}")
    print(f"使用的系统提示词: \"{SYSTEM_PROMPT}\"")
    
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
            print(f"\n[诊断信息]: 正在尝试连接 OpenAI API: {model_name}...")
            
            # --- 核心修改点 2: 在线模型也使用同一个系统提示词 ---
            stream = client.chat.completions.create(
                model=model_name,
                messages=[
                    # 使用共享的系统提示词
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': prompt}
                ],
                stream=True,
                temperature=0.7,
            )
            # ----------------------------------------------------

            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    print(content, end="", flush=True)
                    full_sentence += content
                    sentences = re.split(r'(?<=[。！？\!\?])\s*', full_sentence)
                    
                    if len(sentences) > 1:
                        for sentence in sentences[:-1]:
                            if sentence.strip():
                                text_queue.put(sentence.strip())
                        full_sentence = sentences[-1]

        except openai.APIConnectionError as e:
            print(f"\n无法连接到服务器: {e.__cause__}")
            continue
        except openai.RateLimitError as e:
            print(f"\nAPI 请求已达速率限制: {e.response.text}")
            continue
        except openai.APIStatusError as e:
            print(f"\nAPI 返回非 2xx 状态码: Status {e.status_code}, Response: {e.response}")
            continue
        except Exception as e:
            print(f"\n调用 OpenAI API 时发生未知错误: {e}")
            continue

        if full_sentence.strip():
            text_queue.put(full_sentence.strip())
        print()