import threading
import time
import cv2
import queue
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import Label, Frame, Button, filedialog
from frcnn import FRCNN

class CameraReader(threading.Thread):
    def __init__(self, frame_queue, video_source):
        super().__init__()
        self.frame_queue = frame_queue
        self.video_source = video_source
        self.daemon = True

    def run(self):
        cap = cv2.VideoCapture(self.video_source)

        if not cap.isOpened():
            print("Failed to open camera!")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame!")
                break

            self.frame_queue.put(frame)  # 将摄像头图像传给队列
            time.sleep(0.03)  # 控制帧率

        cap.release()

class ImageProcessor(threading.Thread):
    def __init__(self, frame_queue, result_queue):
        super().__init__()
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.daemon = True
        self.frcnn = FRCNN()

    def run(self):
        while True:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                processed_frame, clses = self.process_image(frame)
                self.result_queue.put(processed_frame)  # 将处理后的图像传给结果队列
                self.result_queue.put(clses)  # 将处理后的图像的类别传给结果队列

    def process_image(self, image):
        print(f"Processing image of shape: {image.shape}")
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(np.uint8(frame))

        t1 = time.time()
        #rcnn图像处理，F12查看代码
        frame , clses = self.frcnn.detect_image(frame)
        #print(clses)
        frame = np.array(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        fps = (1 + (1. / (time.time() - t1))) / 2
        print("fps= %.2f" % (fps))
        frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame, clses

class GuiUpdater:
    def __init__(self, result_queue, frame_queue):
        self.result_queue = result_queue
        self.frame_queue = frame_queue
        self.show_camera = True
        self.clses = []
        self.old_clses = []
        self.data = []
        self.runtime = 0
        self.categories = ['CA001','CA002','CA003','CA004','CB001','CB002','CB003','CB004','CC001','CC002','CC003','CC004','CD001','CD002','CD003','CD004']

        self.root = tk.Tk()
        self.root.title("Video Processing")

        # 猪画布
        self.main_frame = Frame(self.root)
        self.main_frame.pack()
        # 画布显示图像
        self.canvas = tk.Canvas(self.main_frame, width=360, height=640)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 信息显示区
        self.info_frame = Frame(self.main_frame, width=400, height=540, bg="gray")
        self.info_frame.pack(side=tk.RIGHT, padx=10, fill=tk.Y, anchor="n")

        self.create_info_labels()
        # 创建按钮和状态区
        self.button_frame = Frame(self.root)
        self.button_frame.pack(side=tk.BOTTOM, anchor='se' ,pady=10)
        # 状态标签
        self.status_label = Label(self.button_frame, text="空闲", font=('gothic', 20), fg=rgb_to_hex((255, 0, 0)))
        self.status_label.pack(side=tk.LEFT, padx=10)
        # 开始按钮
        self.start_button = Button(self.button_frame, text="开始", command=self.start_processing, width=10, height=2, font=('gothic', 20))
        self.start_button.pack(side=tk.LEFT, padx=10)
        # 开始更新界面
        self.update_image()

    def update_image(self):
        if self.runtime > 8:
            self.show_camera = True  # 切换到摄像头图像, 结束
            self.status_label.configure(text="结束",fg=rgb_to_hex((255, 0, 0)))
        #接收线程数据
        if not self.result_queue.empty() and not self.show_camera:
            self.runtime += 1
            frame = self.result_queue.get()
            self.clses = self.result_queue.get()
            print(self.clses)   # 获取处理后的图像
        elif not self.frame_queue.empty() and self.show_camera:
            frame = self.frame_queue.get()  # 获取摄像头图像
        else:
            frame = None

        #显示数据处理
        alldata = []
        if len(self.clses) > 0 and self.clses != self.old_clses:
            self.old_clses = self.clses
            count = {}
            for elem in self.clses:
                count[elem] = count.get(elem, 0) + 1

            # 将字典转换为嵌套数组
            count = [[key, value] for key, value in count.items()]
            #print(count)
            for j, row in enumerate(count):
                onedata = [f"目标id: {row[0]}", f"目标数量: {row[1]}"]
                alldata.append(onedata)

            self.data = alldata
            #print(self.data)
            # 更新信息标签
            self.create_info_labels()
        # 显示图像
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (360, 640))   # 转换颜色格式
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.image = imgtk  # 保存图像引用，避免被垃圾回收

        self.root.after(30, self.update_image)  # 每30毫秒更新一次图像

    def start_processing(self):
        self.show_camera = False  # 切换到处理后的图像
        self.status_label.configure(text="识别中",fg=rgb_to_hex((255, 0, 0)))
        print("Processing started...")

        
    def create_info_labels(self):
        t1 = time.time()   
        # 检查是否已创建标题标签，如果没有，则创建
        if not hasattr(self, 'title_labels'):
            self.titles = ['识别结果输出区']
            self.title_labels = []  # 存储标题标签

            # 创建并显示标题标签
            for i, title in enumerate(self.titles):
                title_label = Label(self.info_frame, text=title, font=('gothic', 20, "bold"), fg=rgb_to_hex((200, 100, 0)), bg="gray")
                title_label.grid(row=0, column=i, sticky="nw")
                self.title_labels.append(title_label)  # 存储标题标签
            self.info_frame.columnconfigure(len(self.titles), minsize=120)
        # 超过20条数据，则删除最早的数据，循环显示最新的数据    
        if len(self.data) >= 20:  
            # 清空之前的数据标签
            for widget in self.info_frame.winfo_children():
                if widget not in self.title_labels:  # 确保不删除标题标签
                    widget.destroy()

            # 超过删除
            self.data.pop(0)

            # 添加数据标签
            for j, row in enumerate(self.data):
                for i, item in enumerate(row):
                    data_label = Label(self.info_frame, text=item, font=('gothic', 12), fg=rgb_to_hex((255, 0, 255)), bg="gray")
                    data_label.grid(row=j + 1, column=i)
        else:
            for widget in self.info_frame.winfo_children():
                if widget not in self.title_labels:  # 确保不删除标题标签
                    widget.destroy()
            print(self.data)
            for j, row in enumerate(self.data):
                for i, item in enumerate(row):
                    data_label = Label(self.info_frame, text=item, font=('gothic', 12), fg=rgb_to_hex((25, 0, 2)), bg="gray")
                    data_label.grid(row=j + 1, column=i, sticky="w", padx=1)

        t2 = time.time()
        print(f"更新数据耗时：{t2 - t1}秒")

    def run(self):
        self.root.mainloop()

def rgb_to_hex(rgb):
    """将 RGB 值转换为十六进制颜色代码."""
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

# 主函数
def main():
    frame_queue = queue.Queue()  # 摄像头帧队列
    result_queue = queue.Queue()  # 处理后的图像队列

    # 启动摄像头读取线程
    camera_reader = CameraReader(frame_queue, r"qq1.mp4")#D:\gtrain_1\faster-rcnn-pytorch-chinese\
    camera_reader.start()

    # 启动图像处理线程
    image_processor = ImageProcessor(frame_queue, result_queue)
    image_processor.start()

    # 启动图形界面更新线程
    gui_updater = GuiUpdater(result_queue, frame_queue)
    gui_updater.run()

if __name__ == "__main__":
    main()
