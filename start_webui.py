# start_webui.py

import gradio as gr
import multiprocessing as mp
import queue
import threading

from ollama_client import stream_ollama_response, stream_openai_response 
from tts_converter import convert_text_to_audio
from audio_player import play_audio_data
from config_loader import config

# --- 全局队列 ---
user_input_queue = mp.Queue()
text_to_speech_queue = mp.Queue()
audio_data_queue = mp.Queue()
ui_update_queue = mp.Queue()
player_command_queue = mp.Queue()
tts_command_queue = mp.Queue()

def launch_backend_processes():
    print("正在启动后端服务进程...")
    
    llm_process = None
    system_prompt = config['system_prompt']
    if config['use_online_model']:
        online_config = config['online_model']
        llm_process = mp.Process(target=stream_openai_response, args=(user_input_queue, text_to_speech_queue, online_config, system_prompt, ui_update_queue))
    else:
        local_config = config['local_model']
        llm_process = mp.Process(target=stream_ollama_response, args=(user_input_queue, text_to_speech_queue, local_config, system_prompt, ui_update_queue))
    
    tts_process = mp.Process(target=convert_text_to_audio, args=(text_to_speech_queue, audio_data_queue, tts_command_queue))
    player_process = mp.Process(target=play_audio_data, args=(audio_data_queue, player_command_queue))

    llm_process.daemon = True
    tts_process.daemon = True
    player_process.daemon = True

    llm_process.start()
    tts_process.start()
    player_process.start()
    
    print("后端服务进程已成功启动。")

def handle_user_message(user_input, history):
    """
    处理用户从UI发送的消息，并使用新的 'messages' 格式。
    """
    if not user_input.strip():
        return history, "请输入内容后再发送"

    print(f"[WebUI]: 收到用户输入: {user_input}")
    # 清理可能存在的旧更新
    while not ui_update_queue.empty():
        try:
            ui_update_queue.get_nowait()
        except queue.Empty:
            continue

    # --- 2. 修改点: 使用新的字典格式追加历史记录 ---
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": ""})
    
    # 将用户输入放入队列，由LLM进程处理
    user_input_queue.put(user_input)

    bot_response_content = ""
    # 循环从ui_update_queue获取LLM的流式输出
    while True:
        try:
            update = ui_update_queue.get(timeout=20)
            if update is None:
                break
            bot_response_content += update
            # 更新最后一条消息（也就是助手消息）的内容
            history[-1]['content'] = bot_response_content
            yield history, "AI正在响应..."
        except queue.Empty:
            print("[WebUI]: 等待AI响应超时。")
            break
    
    print(f"[WebUI]: AI响应结束: {bot_response_content}")
    yield history, "AI响应结束"

def terminate_and_clear_audio():
    """
    终止当前播放并清空待播队列。
    """
    print("[WebUI]: 用户点击终止，发送CLEAR命令。")
    player_command_queue.put("CLEAR")
    tts_command_queue.put("CLEAR")
    return "已发送清空命令"

# --- 构建 Gradio Web UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 语音对话 Web UI")
    gr.Markdown("在下方的输入框中输入你的问题，点击发送或按回车。AI的回答将以文本形式显示，并自动转换为语音播放。")

    # --- 1. 修改点: 为Chatbot指定 type='messages' ---
    chatbot = gr.Chatbot(
        label="对话历史", 
        height=500, 
        avatar_images=("./asset/icons/avatar_user.png", "./asset/icons/avatar_bot.jpg"),
        type='messages' 
    )
    status_textbox = gr.Textbox(label="状态", interactive=False)

    with gr.Row():
        msg_textbox = gr.Textbox(placeholder="输入你的问题...", label="用户输入", container=False, scale=7)
        send_button = gr.Button("发送", variant="primary", scale=1)
        terminate_button = gr.Button("清空音频队列", variant="stop", scale=2)

    # 绑定事件
    msg_textbox.submit(handle_user_message, [msg_textbox, chatbot], [chatbot, status_textbox])
    send_button.click(handle_user_message, [msg_textbox, chatbot], [chatbot, status_textbox])
    terminate_button.click(terminate_and_clear_audio, outputs=[status_textbox])
    
    # 清空输入框
    send_button.click(lambda: "", None, msg_textbox)
    msg_textbox.submit(lambda: "", None, msg_textbox)

if __name__ == "__main__":
    launch_backend_processes()
    print("正在启动 Gradio Web UI...")
    demo.launch()
    print("Web UI已关闭，程序结束。")