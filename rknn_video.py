    def process_infrared_images_from_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if folder_path:  # 确保用户选择了一个文件夹
            supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
            all_files = os.listdir(folder_path)
            imageFiles = [file for file in all_files if os.path.splitext(file)[1].lower() in supported_extensions]
            print(imageFiles)  # 或者根据需要处理这些图像文件
            
            RKNN_MODEL    = '/home/firefly/zhgq_tisr/basicTISR/models/rknn/A_ETISR_256X320.rknn'  # RKNN 模型文件路径
            THREADS_NUMS = 3

            pool = rknnPoolExecutor(
                rknnModel=RKNN_MODEL,
                TPEs=THREADS_NUMS,
                func=SISR)
            
            for imageName in imageFiles:
                img_path = os.path.join(folder_path, imageName)
                img = cv2.imread(img_path)

                if img is not None:
                    pool.put(img)
                else:
                    print(f"Warning: Failed to load image {imageName}")

            for imageName in imageFiles:
                (frame, processed_frame), flag = pool.get()
                if flag == False:
                    break

                self.original_image = frame
                self.reconstructed_image = processed_frame
                self.update_images()