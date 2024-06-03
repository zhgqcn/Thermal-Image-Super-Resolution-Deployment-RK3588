from PyQt5.QtCore import pyqtSignal, QObject
import os
import cv2
from PyQt5.QtWidgets import QFileDialog
from rknnlite.api import RKNNLite
from rknnpool import rknnPoolExecutor, SISR

class ImageProcessor(QObject):
    update_images_signal = pyqtSignal(object, object)  # 用于更新图像的信号

    def __init__(self):
        super().__init__()
        self.pool = None

    def process_infrared_images_from_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if folder_path:
            supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
            all_files = os.listdir(folder_path)
            imageFiles = [file for file in all_files if os.path.splitext(file)[1].lower() in supported_extensions]
            
            RKNN_MODEL = '/home/firefly/zhgq_tisr/basicTISR/models/rknn/A_ETISR_256X320.rknn'
            THREADS_NUMS = 3

            self.pool = rknnPoolExecutor(
                rknnModel=RKNN_MODEL,
                TPEs=THREADS_NUMS,
                func=self.SISR)
            
            for imageName in imageFiles:
                img_path = os.path.join(folder_path, imageName)
                img = cv2.imread(img_path)

                if img is not None:
                    self.pool.put((img, imageName))  # 将图像和图像名称一起提交
                else:
                    print(f"Warning: Failed to load image {imageName}")

            self.process_results(imageFiles)

    def process_results(self, imageFiles):
        for _ in imageFiles:
            result = self.pool.get()
            if result is None:
                break
            (frame, processed_frame), flag = result
            if flag:
                self.update_images_signal.emit(frame, processed_frame)  # 发射信号以更新图像

