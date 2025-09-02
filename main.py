# main.py

import multiprocessing as mp
import sys
from ollama_client import stream_ollama_response, stream_openai_response 
from tts_converter import convert_text_to_audio
from audio_player import play_audio_data
from config_loader import config # <--- 导入配置

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
    # 创建队列
    user_input_queue = mp.Queue()
    text_to_speech_queue = mp.Queue()
    audio_data_queue = mp.Queue()

    llm_process = None
    system_prompt = config['system_prompt']

    # 根据配置文件决定启动哪个LLM进程
    if config['use_online_model']:
        print("--- 根据配置，启动联网 OpenAI 模型 ---")
        online_config = config['online_model']
        llm_process = mp.Process(
            target=stream_openai_response, 
            args=(user_input_queue, text_to_speech_queue, online_config, system_prompt)
        )
    else:
        print("--- 根据配置，启动本地 Ollama 模型 ---")
        local_config = config['local_model']
        llm_process = mp.Process(
            target=stream_ollama_response, 
            args=(user_input_queue, text_to_speech_queue, local_config, system_prompt)
        )

    # TTS 转换进程
    tts_conversion_process = mp.Process(
        target=convert_text_to_audio, 
        args=(text_to_speech_queue, audio_data_queue)
    )
    
    # 音频播放进程
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