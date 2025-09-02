import sys

def capture_input(input_queue):
    """
    在终端监听用户输入，并将每行输入放入队列。
    以 'exit' 或 'quit' 结束。
    """
    print("你好！请输入你的问题。输入 'exit' 或 'quit' 来结束对话。")
    while True:
        try:
            line = sys.stdin.readline().strip()
            if line.lower() in ["exit", "quit"]:
                print("程序退出。")
                input_queue.put(None) 
                break
            if line:
                input_queue.put(line)
        except KeyboardInterrupt:
            print("\n检测到中断，程序退出。")
            input_queue.put(None)
            break