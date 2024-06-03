from queue import Queue
from rknnlite.api import RKNNLite
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np

def GuidedSR(rknn_lite, ori_img):
    img_thermal, img_visible = ori_img

    img_the = cv2.cvtColor(img_thermal, cv2.COLOR_BGR2RGB)[:, :, 0]
    height, width = img_the.shape[:2]
    img_the = cv2.resize(img_the, (width * 4, height * 4), interpolation=cv2.INTER_LINEAR)
    img_the = np.clip(img_the, 0, 255).astype(np.uint8)
    img_the = np.expand_dims(img_the, axis=0)
    img_the = np.expand_dims(img_the, axis=0)

    img_rgb = cv2.cvtColor(img_visible, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_rgb = np.expand_dims(img_rgb, axis=0)
    img_rgb = np.expand_dims(img_rgb, axis=0)

    outputs = rknn_lite.inference(inputs=[img_rgb, img_the])
    output_img = outputs[0].squeeze(0).squeeze(0) * 255
    output_img = np.clip(output_img, 0, 255).astype(np.uint8)

    # 在界面上显示原始图像和重建结果图像
    ori_thermal = cv2.resize(img_thermal, (640, 480))
    out_thermal = cv2.resize(output_img, (640, 480))

    ori_thermal = cv2.cvtColor(ori_thermal, cv2.COLOR_BGR2RGB)
    out_thermal = cv2.cvtColor(out_thermal, cv2.COLOR_GRAY2RGB)

    return [ori_thermal, out_thermal]


def SISR(rknn_lite, ori_img):
    img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)[:, :, 0]
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)

    outputs = rknn_lite.inference(inputs=[img])
    output_img = outputs[0].squeeze(0).squeeze(0) * 255
    output_img = np.clip(output_img, 0, 255).astype(np.uint8)

    # 在界面上显示原始图像和重建结果图像
    ori_img = cv2.resize(ori_img, (640, 480))
    output_img = cv2.resize(output_img, (640, 480))

    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    output_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2RGB)

    return [ori_img, output_img]

def initRKNN(rknnModel=" ", id=0):
    rknn_lite = RKNNLite()
    ret = rknn_lite.load_rknn(rknnModel)
    if ret != 0:
        print("Load RKNN rknnModel failed")
        exit(ret)
    if id == 0:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    elif id == 1:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
    elif id == 2:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_2)
    elif id == -1:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    else:
        ret = rknn_lite.init_runtime()
    if ret != 0:
        print("Init runtime environment failed")
        exit(ret)
    print(rknnModel, "\t\tdone")
    return rknn_lite
 
 
def initRKNNs(rknnModel=" ", TPEs=1):
    rknn_list = []
    for i in range(TPEs):
        rknn_list.append(initRKNN(rknnModel, i % 3))
    return rknn_list
 
 
class rknnPoolExecutor():
    def __init__(self, rknnModel, TPEs, func):
        self.TPEs = TPEs
        self.queue = Queue()
        self.rknnPool = initRKNNs(rknnModel, TPEs)
        self.pool = ThreadPoolExecutor(max_workers=TPEs)
        self.func = func
        self.num = 0
 
    def put(self, frame):
        # 将原图像和处理任务作为一个元组提交
        future = self.pool.submit(self.func, self.rknnPool[self.num % self.TPEs], frame)
        self.queue.put((frame, future))
        self.num += 1
 
    def get(self):
        if self.queue.empty():
            return None, False
        # 从队列中获取原图像和处理任务的元组
        frame, future = self.queue.get()
        # 等待任务完成并获取处理后的图像
        processed_frame = future.result()
        # 返回原图像和处理后的图像的成对
        return (frame, processed_frame), True
 
    def release(self):
        self.pool.shutdown()
        for rknn_lite in self.rknnPool:
            rknn_lite.release()

class rknnPoolExecutorVideos():
    def __init__(self, rknnModel, TPEs, func):
        self.TPEs = TPEs
        self.queue = Queue()
        self.rknnPool = initRKNNs(rknnModel, TPEs)
        self.pool = ThreadPoolExecutor(max_workers=TPEs)
        self.func = func
        self.num = 0

    def put(self, frame):
        self.queue.put(self.pool.submit(
            self.func, self.rknnPool[self.num % self.TPEs], frame))
        self.num += 1

    def get(self):
        if self.queue.empty():
            return None, False
        fut = self.queue.get()
        return fut.result(), True

    def release(self):
        self.pool.shutdown()
        for rknn_lite in self.rknnPool:
            rknn_lite.release()
    

class rknnPoolExecutorVideosGuided():
    def __init__(self, rknnModel, TPEs, func):
        self.TPEs = TPEs
        self.queue = Queue()
        self.rknnPool = initRKNNs(rknnModel, TPEs)
        self.pool = ThreadPoolExecutor(max_workers=TPEs)
        self.func = func
        self.num = 0

    def put(self, frame):
        self.queue.put(self.pool.submit(self.func, self.rknnPool[self.num % self.TPEs], frame))
        self.num += 1

    def get(self):
        if self.queue.empty():
            return None, False
        fut = self.queue.get()
        return fut.result(), True

    def release(self):
        self.pool.shutdown()
        for rknn_lite in self.rknnPool:
            rknn_lite.release()