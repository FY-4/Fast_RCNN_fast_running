import threading
import time
import cv2
import queue
import numpy as np
from PIL import Image
from frcnn import FRCNN

# 图片处理函数（这里只是一个简单的示例）
def process_image(image, frcnn):
    # 这里你可以添加任何图像处理代码，当前代码只是打印图像的形状
    print(f"Processing image of shape: {image.shape}")
    #image = cv2.resize(image, (960, 540))  # 缩放图像
    frame = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # 转变成Image
    frame = Image.fromarray(np.uint8(frame))
    # 进行检测

    t1 = time.time()
    frame = np.array(frcnn.detect_image(frame))
    # RGBtoBGR满足opencv显示格式
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    fps  = ( 1 + (1./(time.time()-t1)) ) / 2
    print("fps= %.2f"%(fps))
    frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Processed Image", frame)
    if cv2.waitKey(200) & 0xFF == ord('q'):  # 按下 q 键退出
    # 假设在处理后会有一些延迟
        cv2.destroyAllWindows()

# 摄像头读取线程
def read_from_camera(frame_queue):
    # 打开摄像头（通常是0，如果有多个摄像头，可以使用其他编号）
    cap = cv2.VideoCapture(r"D:\gtrain_1\faster-rcnn-pytorch-chinese\qq1.mp4")

    if not cap.isOpened():
        print("Failed to open camera!")
        return

    timeb = 0
    while True:
        # 读取摄像头中的一帧图像
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame!")
            break

        # 每两秒传输一帧图像给处理线程
        if timeb == 0:
            t1 = time.time()
        if time.time() - t1 > 1.8:
            frame_queue.put(frame)
            print("Sent a new frame to processing thread.")
            timeb = 0
        else:
            timeb = 1
        time.sleep(0.03)  # 控制帧率
        # 等待2秒后发送下一帧

    cap.release()

# 图像处理线程
def process_thread(frame_queue):
    frcnn = FRCNN()
    while True:
        # 获取一张图像进行处理
        if not frame_queue.empty():
            frame = frame_queue.get()
            # 处理图像
            process_image(frame, frcnn)
        else:
            # 如果队列为空，稍微等待一下
            time.sleep(0.1)

# 主函数
def main():
    # 创建一个队列，用于线程间传递图像
    frame_queue = queue.Queue()

    # 创建并启动摄像头读取线程
    camera_thread = threading.Thread(target=read_from_camera, args=(frame_queue,))
    camera_thread.daemon = True  # 设置为守护线程，程序退出时自动结束
    camera_thread.start()

    # 创建并启动图像处理线程
    processing_thread = threading.Thread(target=process_thread, args=(frame_queue,))
    processing_thread.daemon = True
    processing_thread.start()

    # 主线程保持运行，直到用户按下 Ctrl+C
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Program interrupted. Exiting...")
        exit(0)

if __name__ == "__main__":
    main()
