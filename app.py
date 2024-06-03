import sys, os, cv2, time
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QToolBar, QLabel, QVBoxLayout, QWidget, QSplitter
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QFileDialog
from rknnpool import rknnPoolExecutor, SISR, rknnPoolExecutorVideos, GuidedSR, rknnPoolExecutorVideosGuided

import numpy as np
from rknnlite.api import RKNNLite
# os.environ["QT_QPA_PLATFORM"] = "wayland"

rknn_lite = RKNNLite()
ret = rknn_lite.load_rknn("model/A_ETISR_120X160_NoQuan.rknn")
ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

rknn_lite_guided = RKNNLite()
ret_guided = rknn_lite_guided.load_rknn("model/A_guided_GTISR_OUR_120X160_NoQuan.rknn")
ret_guided = rknn_lite_guided.init_runtime(core_mask=RKNNLite.NPU_CORE_0)


class InfraredSuperResolutionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.active_button = None  # 用于跟踪当前激活的按钮

        self.setWindowTitle("红外图像超分辨应用")
        self.setGeometry(0, 0, 1366, 768)  # 窗口大小适应显示器屏幕大小

        # 创建工具栏
        toolbar = QToolBar()
        self.addToolBar(toolbar)

        single_infrared_sr_button = QPushButton("单幅红外SR")
        single_infrared_sr_button.clicked.connect(self.show_infrared_image_upload)
        toolbar.addWidget(single_infrared_sr_button)

        guided_infrared_sr_button = QPushButton("引导红外SR")
        guided_infrared_sr_button.clicked.connect(self.show_infrared_visible_image_upload)
        toolbar.addWidget(guided_infrared_sr_button)

        continuous_single_infrared_sr_button = QPushButton("连续单幅红外SR")
        continuous_single_infrared_sr_button.clicked.connect(self.process_infrared_images_from_folder)
        toolbar.addWidget(continuous_single_infrared_sr_button)

        continuous_guided_infrared_sr_button = QPushButton("连续引导红外SR")
        continuous_guided_infrared_sr_button.clicked.connect(self.process_infrared_visible_images_from_folder)
        toolbar.addWidget(continuous_guided_infrared_sr_button)

        # 初始化界面
        self.layout = QVBoxLayout()
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.central_widget.setLayout(self.layout)

        # 创建QSplitter用于左右布局
        self.splitter = QSplitter(self)
        self.layout.addWidget(self.splitter)

        # 显示原始图像和重建结果图像的 QLabel
        self.original_img_label = QLabel()
        self.reconstructed_img_label = QLabel()

        self.original_img_label.setScaledContents(False)
        self.reconstructed_img_label.setScaledContents(False)

        # 将 QLabel 添加到 QSplitter 中
        self.splitter.addWidget(self.original_img_label)
        self.splitter.addWidget(self.reconstructed_img_label)

        # 设置QSplitter分隔线的初始位置，0表示左侧占比，100表示右侧占比
        self.splitter.setSizes([50, 50])  # 左右各占一半空间

        # 按钮初始化和点击事件连接代码...
        single_infrared_sr_button.clicked.connect(lambda: self.change_button_color(single_infrared_sr_button))
        guided_infrared_sr_button.clicked.connect(lambda: self.change_button_color(guided_infrared_sr_button))
        continuous_single_infrared_sr_button.clicked.connect(lambda: self.change_button_color(continuous_single_infrared_sr_button))
        continuous_guided_infrared_sr_button.clicked.connect(lambda: self.change_button_color(continuous_guided_infrared_sr_button))

    def change_button_color(self, button):
        if self.active_button and self.active_button != button:
            # 恢复之前激活按钮的默认样式
            self.active_button.setStyleSheet("")
        # 改变当前按钮的颜色并更新当前激活的按钮
        button.setStyleSheet('QPushButton {background-color: #00FF00; color: black;}')
        self.active_button = button

    def update_image_in_label(self, label, img):
        new_width = 640  # 图像宽度
        new_height = 512  # 图像高度

        img = cv2.resize(img, (new_width, new_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        q_img = QImage(img.data, new_width, new_height, img.shape[2] * new_width, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(q_img))
        
    def update_images(self):
        self.update_image_in_label(self.original_img_label, self.original_image)
        self.update_image_in_label(self.reconstructed_img_label, self.reconstructed_image)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_images()
    
    # 单幅红外SR
    def show_infrared_image_upload(self):
        file_dialog = QFileDialog()
        
        if file_dialog.exec_():
            img_path = file_dialog.selectedFiles()[0]
            ori_img = cv2.imread(img_path)
            img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)[:, :, 0]

            # 处理图像
            img = np.expand_dims(img, axis=0)
            img = np.expand_dims(img, axis=0)

            # 进行推理
            outputs = rknn_lite.inference(inputs=[img])

            output_img = outputs[0].squeeze(0).squeeze(0) * 255
            output_img = np.clip(output_img, 0, 255).astype(np.uint8)

            # 在界面上显示原始图像和重建结果图像
            ori_img = cv2.resize(ori_img, (640, 480))
            output_img = cv2.resize(output_img, (640, 480))

            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
            output_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2RGB)

            # 在界面上显示原始图像和重建结果图像
            self.original_image = ori_img
            self.reconstructed_image = output_img
            self.update_images()

    # 引导红外SR
    def show_infrared_visible_image_upload(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        if file_dialog.exec_():
            file_paths = file_dialog.selectedFiles()
            # 确保用户至少选择了两个文件
            if len(file_paths) >= 2:
                # 获取前两个文件的路径
                img_path1, img_path2 = file_paths[:2]

                ori_img = cv2.imread(img_path1)
                img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)[:, :, 0]
                height, width = img.shape[:2]
                upsampled_img = cv2.resize(img, (width * 4, height * 4), interpolation=cv2.INTER_LINEAR)
                img = np.clip(upsampled_img, 0, 255).astype(np.uint8)
        
                rgb_img = cv2.imread(img_path2)
                rgb = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
                rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            
                # 处理图像
                img = np.expand_dims(img, axis=0)
                img = np.expand_dims(img, axis=0)

                rgb = np.expand_dims(rgb, axis=0)
                rgb = np.expand_dims(rgb, axis=0)
                
                # 进行推理
                outputs = rknn_lite_guided.inference(inputs=[rgb, img])

                output_img = outputs[0].squeeze(0).squeeze(0) * 255
                output_img = np.clip(output_img, 0, 255).astype(np.uint8)
              
                # 在界面上显示原始图像和重建结果图像
                self.original_image = ori_img
                self.reconstructed_image = output_img
                self.update_images()

            else:
                print("Please select at least two images.")

    # 连续单幅红外SR
    def process_infrared_images_from_folder(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("Videos (*.mp4)")
        if file_dialog.exec_():
            video_file = file_dialog.selectedFiles()[0]
        else:
            print('No file selected.')
            exit(-1)

        cap = cv2.VideoCapture(video_file)
        RKNN_MODEL    = 'model/A_ETISR_256X320.rknn'  # RKNN 模型文件路径
        THREADS_NUMS = 3

        pool = rknnPoolExecutorVideos(
            rknnModel=RKNN_MODEL,
            TPEs=THREADS_NUMS,
            func=SISR)
        
        # 初始化异步所需要的帧
        if (cap.isOpened()):
            for i in range(THREADS_NUMS + 1):
                ret, frame = cap.read()
                if not ret:
                    cap.release()
                    del pool
                    exit(-1)
                pool.put(frame)

        frames, loopTime, initTime = 0, time.time(), time.time()
        while (cap.isOpened()):
            frames += 1
            ret, frame = cap.read()
            if not ret:
                break
            pool.put(frame)
            frame, flag = pool.get()
            if flag == False:
                break
            # cv2.imshow('test', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # 在界面上显示原始图像和重建结果图像
            self.original_image = frame[0]
            self.reconstructed_image = frame[1]
            self.update_images()

    def process_infrared_visible_images_from_folder(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        if file_dialog.exec_():
            file_paths = file_dialog.selectedFiles()
            # 确保用户至少选择了两个文件
            if len(file_paths) >= 2:
                # 获取前两个文件的路径
                video_thermal, video_visible = file_paths[:2]

                cap_thermal = cv2.VideoCapture(video_thermal)
                cap_visible = cv2.VideoCapture(video_visible)

                RKNN_MODEL_Guided = 'model/A_guided_GTISR_OUR_120X160_NoQuan.rknn'  # RKNN 模型文件路径
                THREADS_NUMS = 3

                pool_Guided = rknnPoolExecutorVideosGuided(
                    rknnModel=RKNN_MODEL_Guided,
                    TPEs=THREADS_NUMS,
                    func=GuidedSR)

                # 初始化异步所需要的帧
                if (cap_thermal.isOpened() and cap_visible.isOpened()):
                    for i in range(THREADS_NUMS + 1):
                        ret_thermal, frame_thermal = cap_thermal.read()
                        ret_visible, frame_visible = cap_visible.read()
                        if not ret_thermal or not ret_visible:
                            cap_thermal.release()
                            cap_visible.release()
                            del pool_Guided
                            exit(-1)
                        pool_Guided.put([frame_thermal, frame_visible])

                frames, loopTime, initTime = 0, time.time(), time.time()
                while (cap_thermal.isOpened() and cap_visible.isOpened()):
                    frames += 1
                    ret_thermal, frame_thermal = cap_thermal.read()
                    ret_visible, frame_visible = cap_visible.read()

                    if not ret_thermal or not ret_visible:
                        break

                    pool_Guided.put([frame_thermal, frame_visible])
                    frame, flag = pool_Guided.get()
                    if flag == False:
                        break
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    # 在界面上显示原始图像和重建结果图像
                    self.original_image = frame[0]
                    self.reconstructed_image = frame[1]
                    self.update_images()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    infrared_app = InfraredSuperResolutionApp()
    infrared_app.show()
    sys.exit(app.exec_())
