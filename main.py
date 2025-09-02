# main.py

import multiprocessing as mp
import sys
from ollama_client import stream_ollama_response, stream_openai_response 
# 导入修改后的转换器和新的播放器
from tts_converter import convert_text_to_audio
from audio_player import play_audio_data

def get_openai_config():
    base_url = "https://api.deepseek.com"
    model_name = "deepseek-chat"
    api_key = "sk-8352c67e63394c44b7401ea73d933d07"
    return model_name, api_key, base_url

def get_ollama_config():
    print("\n--- 配置本地 Ollama 模型 ---")
    model_name = input("请输入要使用的本地 Ollama 模型名称 (例如 qwen2:7b): ")
    return model_name

def main_input_loop(input_queue):
    """
    在主进程中处理用户输入。
    """
    print("\n你好！请输入你的问题。输入 'exit' 或 'quit' 来结束对话。")
    for line in sys.stdin:
        line = line.strip()
        if line.lower() in ["exit", "quit"]:
            print("程序退出指令已发送。")
            input_queue.put(None)
            break
        if line:
            input_queue.put(line)

if __name__ == "__main__":
    choice = ""
    while choice not in ["1", "2"]:
        choice = input("请选择要使用的模型后端:\n1. 本地 Ollama\n2. 联网 OpenAI 协议模型\n请输入选项 (1 或 2): ")

    # 创建两个队列用于进程间通信
    user_input_queue = mp.Queue()
    text_to_speech_queue = mp.Queue()
    # 新增一个音频数据队列
    audio_data_queue = mp.Queue()

    llm_process = None

    if choice == "1":
        ollama_model_name = get_ollama_config()
        llm_process = mp.Process(
            target=stream_ollama_response, 
            args=(user_input_queue, text_to_speech_queue, ollama_model_name)
        )
    else: # choice == "2"
        model, key, url = get_openai_config()
        llm_process = mp.Process(
            target=stream_openai_response, 
            args=(user_input_queue, text_to_speech_queue, model, key, url)
        )

    # 进程1: TTS 转换 (原 tts_process)
    tts_conversion_process = mp.Process(
        target=convert_text_to_audio, 
        args=(text_to_speech_queue, audio_data_queue)
    )
    
    # 进程2: 音频播放
    player_process = mp.Process(
        target=play_audio_data, 
        args=(audio_data_queue,)
    )

    # 启动所有后台进程
    if llm_process:
        llm_process.start()
    tts_conversion_process.start()
    player_process.start()

    # 在主进程中运行输入循环
    try:
        main_input_loop(user_input_queue)
    except KeyboardInterrupt:
        print("\n检测到中断，正在关闭程序...")
        user_input_queue.put(None)

    # 等待所有后台进程结束
    if llm_process:
        llm_process.join()
    tts_conversion_process.join()
    player_process.join()

    print("所有进程已结束，程序关闭。")