# -*- coding:utf-8 -*-
import os
import sys
import threading
from functools import wraps
import numpy as np
import time
import cv2
import pickle
import imutils
from PyQt5 import QtWidgets
from PyQt5.QtCore import QCoreApplication, QThread, pyqtSignal, QTimer, QDateTime
from PyQt5.QtGui import QPixmap, QIcon, QImage
from PyQt5.QtWidgets import QApplication, QProgressBar, QStyle, QVBoxLayout, QLabel
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QWidget, QInputDialog

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from datetime import datetime
# 导入自定义包
from ui.WindowUI import Ui_MainWindow
from utils import GeneratorModel
from ui.InfoUI import Ui_Form

# ---------------------- #
# 如果摄像头打开黑屏
# 尝试关闭杀毒软件！！！！！！
# ---------------------- #
# self.cam_id = 0表示调用默认的摄像头，如果笔记本外接了USB摄像头，可以设置为self.cam_id = 1
# ---------------------- #
# 关于人脸图片数量，测试单人100张以上效果比价好，也看到使用SVM建议300张的。手动拍比较麻烦，可以设置自动拍摄，或者通过图像增强的方法。


# progress bar class
class ProBar(QThread):
    bar_signal = pyqtSignal(int)

    def __int__(self):
        super(ProBar, self).__init__()

    def run(self):
        for i in range(100 + 1):
            # time.sleep(0.05)
            self.bar_signal.emit(i)


# 线程
class WorkThread(QThread):
    trigger = pyqtSignal(str)

    def __init__(self):
        super(WorkThread, self).__init__()

    def run(self):
        self.trigger.emit("")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        # 继承ui窗口类
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.retranslateUi(self)

        # ####################### 需更改路径 ######################
        # 导入opencv人脸检测xml文件
        self.cascade = './haar/haarcascade_frontalface_default.xml'
        # 初始化label显示的(黑色)背景
        self.bkg_pixmap = QPixmap('./imgs/bkg.png')
        # OpenCV深度学习人脸检测器的路径
        self.detector_path = "./face_detection_model"
        # OpenCV深度学习面部嵌入模型的路径
        self.embedding_model = "./face_detection_model/openface_nn4.small2.v1.t7"
        # 训练模型以识别面部的路径
        self.recognizer_path = "./saved_weights/recognizer.pickle"
        # 标签编码器的路径
        self.le_path = "./saved_weights/le.pickle"

        # ####################### 窗口初始化 ######################
        # 设置图片背景
        self.ui.label_camera.setPixmap(self.bkg_pixmap)
        # 设置窗口名称和图标
        self.setWindowTitle("FR-Sys")
        self.setWindowIcon(QIcon('./imgs/xdu.jpg'))

        # # QTimer可能导致卡顿
        # # 设置显示日期和时间
        # 初始化一个定时器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.show_time_text)
        # 定义时间任务是一次性任务
        # self.timer.setSingleShot(True)
        # 启动时间任务
        self.timer.start()

        # ####################### 定义标识符 ######################
        # 被调用摄像头的id
        self.cam_id = 0
        # try:
        #     print("Starting initialize Camera···")
        #     # 初始化摄像头
        #     # self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        #     self.cap = cv2.VideoCapture(1)
        #     # 设置显示分辨率和FPS，否则很卡
        #     self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
        #     self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
        #     self.cap.set(cv2.CAP_PROP_FPS, 30)
        # except EnvironmentError as e:
        #     print("Failed to initialized Camera!", e)
        # finally:
        #     print("Finished initialize Camera!")

        # 定义每个人需要采集的人脸数
        self.num_photos_of_person = 30
        # 定义打开相机标识符
        self.flag_open_camera = False
        # 设置相机打开和关闭按键的初始状态
        self.ui.pb_openCamera.setEnabled(True)
        self.ui.pb_closeCamera.setEnabled(False)
        self.ui.pb_startCamRec.setEnabled(False)
        self.ui.pb_stopFace.setEnabled(False)

        # 定义打开文件类型标识符
        self.flag_open_file_type = None
        # 获取当前文件路径
        self.cwd = os.getcwd()

        # progress bar
        # self.ui.progressBar.setGeometry(0, 0, 300, 25)
        self.ui.progressBar.setValue(0)
        # self.ui.progressBar.setMaximum(100)

        # #################### 定义按键连接函数 ####################
        # 开始摄像头识别按键
        self.ui.pb_startCamRec.clicked.connect(self.check_cam_is_open)
        # self.ui.pb_startCamRec.clicked.connect(self.auto_control)
        # 训练模型按键
        self.ui.pb_trainModel.clicked.connect(self.train_model)
        # 设置“退出系统”按键事件, 按下之后退出主界面
        self.ui.pb_exit.clicked.connect(QCoreApplication.instance().quit)
        # 保证图片视频摄像头三者中当且仅有一种类型可以打开，设置三个标识符和槽函数配置切换状态
        # 图片选项 Radio button
        self.ui.rb_openImage.clicked.connect(self.rb_open_image)
        # 视频选项 Radio button
        self.ui.rb_openVideo.clicked.connect(self.rb_open_video)
        # 摄像头选项 Radio button
        self.ui.rb_openCamera.clicked.connect(self.rb_open_camera)
        # 打开图片/视频按键连接函数
        self.ui.pb_openFile.clicked.connect(self.pb_open_file)
        # 打开摄像头按键连接函数
        # 链接到pb_open_file判断标志位后，再启动线程/进程
        self.ui.pb_openCamera.clicked.connect(self.pb_open_file)
        # 关闭摄像头按键连接函数
        self.ui.pb_closeCamera.clicked.connect(self.pb_close_camera)
        # 关闭视频按键连接函数
        self.ui.pb_closeFile.clicked.connect(self.pb_close_file)
        # 关闭人脸识别按键连接函数
        self.ui.pb_stopFace.clicked.connect(self.pb_stop_face)

        # 设置ratio button为unchecked
        self.rb_group = QtWidgets.QButtonGroup()
        self.rb_group.addButton(self.ui.rb_openImage)
        self.rb_group.addButton(self.ui.rb_openVideo)
        self.rb_group.addButton(self.ui.rb_openCamera)

        # #################### 线程 ####################
        # 创建一个关闭摄像头的线程事件，并设置为未触发
        self.th_camera_close = threading.Event()
        self.th_camera_close.clear()
        # 创建一个关闭视频的线程事件，并设置为未触发
        self.th_video_close = threading.Event()
        self.th_video_close.clear()
        # 创建一个关闭人脸识别的线程事件，并设置为未触发
        self.th_face_reg_close = threading.Event()
        self.th_face_reg_close.clear()

    # ############################################################
    # 阻塞！！！！！！
    # 显示系统时间以及相关文字提示函数
    def show_time_text(self):
        # 设置宽度
        self.ui.label_time.setFixedWidth(200)
        # 设置显示文本格式
        self.ui.label_time.setStyleSheet(
            # "QLabel{background:white;}" 此处设置背景色
            "QLabel{color:rgb(0, 0, 0); font-size:14px; font-weight:bold; font-family:宋体;}"
            "QLabel{font-size:14px; font-weight:bold; font-family:宋体;}")
        datetime = QDateTime.currentDateTime().toString()
        self.ui.label_time.setText("" + datetime)

    def pb_close_camera(self):
        """
        关闭相机按键的槽函数，用于触发关闭摄像头线程Event，以便关闭摄像头
        :return:
        """
        if self.cap.isOpened():
            self.ui.textBrowser.append("[INFO] Killing camera stream...")
            # 触发关闭摄像头线程事件
            self.th_camera_close.set()
        else:
            self.ui.pb_openCamera.setEnabled(True)
            self.ui.pb_closeCamera.setEnabled(False)

    def pb_close_file(self):
        """
        设置关闭视频文件的按键连接函数
        :return:
        """
        self.th_video_close.set()

    def pb_stop_face(self):
        self.th_face_reg_close.set()

    def rb_group_checked(self, status=False):
        """
        Ratio Button Group 用于控制按键的行为
        ----
        :param status:如果为False表示RB控件不可用
        :return:
        """
        # 设置ratio button为unchecked
        self.rb_group.setExclusive(True)  # 表示三个控件只能选一个
        self.ui.rb_openImage.setEnabled(status)
        self.ui.rb_openVideo.setEnabled(status)
        self.ui.rb_openCamera.setEnabled(status)

    def button_useful(in_func):
        """
        装饰器对象，用于为被装饰的函数实现 按键标状态更改，减少冗余代码
        :return: 被装饰的对象
        """

        @wraps(in_func)
        def wrap_func(self, *args, **kwargs):
            # 摄像头打开标志位设为False
            self.flag_open_camera = False
            # 打开文件相关的按键可用
            self.ui.pb_openFile.setEnabled(True)
            self.ui.pb_closeFile.setEnabled(True)
            self.ui.pb_startFileRec.setEnabled(True)
            call_in_func = in_func(self, *args, **kwargs)
            # 摄像头相关按键不可用
            self.ui.pb_openCamera.setEnabled(False)
            self.ui.pb_closeCamera.setEnabled(False)
            self.ui.pb_startCamRec.setEnabled(False)
            return call_in_func

        return wrap_func

    @button_useful
    def rb_open_image(self, *args, **kwargs):
        self.flag_open_file_type = "image"

    @button_useful
    def rb_open_video(self, *args, **kwargs):
        self.flag_open_file_type = "video"

    def rb_open_camera(self):
        """
        Ratio Button 槽函数，主要用于控制按键是否可用状态
        :return:
        """
        self.flag_open_camera = True

        # 摄像头相关按键可用
        self.ui.pb_openCamera.setEnabled(True)
        self.ui.pb_closeCamera.setEnabled(False)

        # 打开文件相关的按键不可用
        self.ui.pb_openFile.setEnabled(False)
        self.ui.pb_closeFile.setEnabled(False)
        self.ui.pb_startFileRec.setEnabled(False)

    # 训练人脸识别模型
    def train_model(self):
        q_message = QMessageBox.information(self, "Tips", "你确定要重新训练模型吗？", QMessageBox.Yes | QMessageBox.No)
        if QMessageBox.Yes == q_message:
            GeneratorModel.Generator()
            GeneratorModel.TrainModel()
            self.ui.textBrowser2.append('[INFO] Models have been trained!')
        else:
            self.ui.textBrowser2.append('[INFO] Cancel train process!')

    def check_cam_is_open(self):
        # 判断摄像头是否打开，如果打开则为true，反之为false
        flag = self.cap.isOpened()
        if not flag:
            self.ui.label_camera.clear()
            # 默认打开Windows系统笔记本自带的摄像头，如果是外接USB，可以将0改成1
            self.cap.open(self.cam_id)
        self.th_face_recognition = threading.Thread(target=self.face_recognition)
        self.th_face_recognition.start()

    def pb_open_file(self):
        """
        打开文件/视频按键的槽函数，用于展示图片 或 初始化播放视频所需的视频流
        :return:
        """
        # 在摄像头未开启的状态下，打开图片或视频文件
        if not self.flag_open_camera:
            # 如果打开文件类型为None，弹出提示信息让用户选择想要打开的文件类型
            if self.flag_open_file_type is None:
                QMessageBox.information(self, "Tips", "Please Choose File Type (Image or Video)!", QMessageBox.Ok)
            # 如果打开文件类型不为None，则根据用户选择的图片或视频做进一步动作
            else:
                if self.flag_open_file_type == "image":
                    # # 选择多种类型的文件函数getOpenFileNames的语法：
                    # self.file_name, self.file_type = QFileDialog.getOpenFileNames(self, 'Choose file',self.cwd,
                    #                                                               'jpg(*.jpg);;png(*.png);;video(*.mp4)')
                    # getOpenFileNames()的返回值file_name返回的是完整的文件路径 [列表]!，
                    # 例如：['D:/QT Project/FR-AttSys/imgs/xdu.jpg']
                    self.file_name, self.file_type = QFileDialog.getOpenFileNames(self, 'Choose file', self.cwd,
                                                                                  'jpg(*.jpg);;png(*.png)')
                    if len(self.file_name) == 0:
                        self.ui.textBrowser2.append("Cancel Image Select!")
                    else:
                        self.ui.textBrowser2.append(self.file_name[0])
                        # 一次识别一张图片
                        self.ui.label_videoFile.setPixmap(QPixmap(self.file_name[0]))
                else:
                    self.file_name, self.file_type = QFileDialog.getOpenFileName(self, 'Choose file', self.cwd, '*.mp4')
                    self.ui.textBrowser2.append(self.file_name)

                    # 如果未选择文件，则不执行打开操作
                    if len(self.file_name) == 0:
                        self.ui.textBrowser2.append("Cancel Video Select!")
                    else:
                        self.cap2 = cv2.VideoCapture(self.file_name)
                        self.fps2 = self.cap2.get(cv2.CAP_PROP_FPS)

                        # 设置显示分辨率和FPS，否则很卡
                        self.cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
                        self.cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)
                        # self.cap2.set(cv2.CAP_PROP_FPS, 24)

                        # 启动新的进程，使用display() 函数逐帧读取视频流
                        self.th_display_video = threading.Thread(target=self.display_video)
                        self.th_display_video.start()

        # 打开摄像头标志位为True，调用display获取主机视频流
        else:
            # 创建一个打开摄像头的线程
            self.th_camera_open = threading.Thread(target=self.display_camera)  # daemon=True
            # # 创建一个打开摄像头的进程
            # self.th_camera_open = multiprocessing.Process(target=self.display_camera(display_type="camera"))
            # 启动打开摄像头线程
            self.th_camera_open.start()

    def display_video(self):
        """
        打开文件/视频按键的槽函数，用于播放视频
        :param kwargs:
        :return:
        """
        self.ui.pb_closeCamera.setEnabled(False)
        self.ui.pb_openCamera.setEnabled(False)
        # 使rb按键失效
        self.rb_group_checked(False)

        # # progressbar
        # self.bar_thread = ProBar()
        # self.bar_thread.bar_signal.connect(self.probar_count_change)
        # self.bar_thread.start()

        while self.cap2.isOpened():
            success, frame = self.cap2.read()
            if success:
                # 逐帧读取视频
                # RGB转BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = imutils.resize(frame, width=500)
                # 将视频文件resize到label的尺寸
                # img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888).scaled(
                #     self.ui.label_camera.width(), self.ui.label_camera.height())
                img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                self.ui.label_videoFile.setPixmap(QPixmap.fromImage(img))
                # 显示FPS
                self.ui.lcd_fps1.display("%d" % self.fps2)

                # # log信息
                # logging.debug("Display Function exit! [%s]", time.ctime())
                # 判断摄像头标识符是否为True，如果为True则等待，否则退出
                if self.flag_open_file_type:
                    cv2.waitKey(1)
                else:
                    cv2.waitKey(int(1000 / self.fps2))

                # 判断视频文件关闭事件是否触发
                if self.th_video_close.is_set():
                    self.ui.textBrowser2.append("[INFO] Start closing Video...")
                    # 关闭事件为触发，清空显示label
                    self.th_video_close.clear()
                    self.ui.label_videoFile.clear()
                    self.ui.lcd_fps1.display(0)
                    self.ui.progressBar.setValue(0)

                    self.ui.textBrowser2.append("[INFO] Succeed to close Video!")
                    break
            else:
                self.cap2.release()
                self.ui.lcd_fps1.display(0)

        # 恢复rb按键
        self.rb_group_checked(True)

    def probar_count_change(self, value):
        # progressbar
        self.ui.progressBar.setValue(value)

    def display_camera(self, **kwargs):
        """
        打开摄像头槽函数，用于控制摄像头的视频流，并带有人脸检测功能！
        :param kwargs:
        :return:
        """
        display_type = kwargs
        self.ui.pb_closeCamera.setEnabled(True)
        self.ui.pb_openCamera.setEnabled(False)
        self.ui.pb_startCamRec.setEnabled(True)
        self.rb_group_checked(False)
        self.ui.label_camera.clear()

        try:
            self.ui.textBrowser.append("[INFO] Start calling camera...")
            # 文本框显示到底部
            self.cursor = self.ui.textBrowser.textCursor()
            self.ui.textBrowser.moveCursor(self.cursor.End)
            # 初始化摄像头
            # self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

            self.cap = cv2.VideoCapture(self.cam_id)
            # 设置显示分辨率和FPS，否则很卡
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

        except EnvironmentError as e:
            self.ui.textBrowser.append("[INFO] Failed to initialize Camera!", e)
        finally:
            self.ui.textBrowser.append("[INFO] End of initializing Camera!")

        # 加载 Haar级联人脸检测库
        detector = cv2.CascadeClassifier(self.cascade)
        self.ui.textBrowser.append("[INFO] Starting camera stream...")

        # webcam 不适用，只适用于视频读取
        # self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        # 实时计算FPS
        # 用来记录处理最后一帧的时间
        prev_frame_time = 0

        # 循环来自视频文件流的帧
        while self.cap.isOpened():
            success, frame = self.cap.read()
            QApplication.processEvents()
            # self.ui.label_camera.width()
            frame2 = imutils.resize(frame, width=600)
            # rects = detector.detectMultiScale(cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), scaleFactor=1.1,
            #                                   minNeighbors=5, minSize=(30, 30))

            # 完成此帧处理的时间
            new_frame_time = time.time()
            denominator = new_frame_time - prev_frame_time
            if denominator <= 0:
                fps = 60
            else:
                fps = 1.0 / (denominator)

            prev_frame_time = new_frame_time
            # converting the fps into integer
            fps = int(fps)

            if fps <= 0:
                fps = 0
            # str: putText function
            # fps_str = "FPS: %.2f" % fps
            fps_str = str(fps)

            # for (x, y, w, h) in rects:
            #     cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #     frame2 = cv2.putText(frame2, "Detecting Faces",
            #                          (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 100, 50), 2)
            # cv2.putText(frame2, fps_str, (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 2)

            # 这里指的是显示原图
            show_video2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            # opencv读取图片的样式，不能通过Qlabel进行显示，需要转换为Qimage。
            # QImage(uchar * data, int width, int height, int bytesPerLine, Format format)
            video_img = QImage(show_video2.data, show_video2.shape[1], show_video2.shape[0], QImage.Format_RGB888)
            self.ui.label_camera.setPixmap(QPixmap.fromImage(video_img))

            # 显示FPS
            self.ui.lcd_fps2.display("%d" % fps)

            # 判断摄像头关闭事件是否触发
            if self.th_camera_close.is_set():
                # 关闭事件为触发，清空显示label
                self.th_camera_close.clear()

                # 恢复用于显示摄像头内容的区域的背景
                self.ui.label_camera.setPixmap(self.bkg_pixmap)
                self.ui.pb_closeCamera.setEnabled(False)
                self.ui.pb_openCamera.setEnabled(True)
                self.ui.pb_startCamRec.setEnabled(False)
                self.ui.pb_stopFace.setEnabled(False)
                self.ui.textBrowser.append("[INFO] Turn off the camera successfully!")
                break

        # 因为最后一张画面会显示在GUI中，此处实现清除。
        self.ui.label_camera.setPixmap(self.bkg_pixmap)
        self.ui.lcd_fps2.display(0)
        self.rb_group_checked(True)

        self.cap.release()
        cv2.destroyAllWindows()

    def face_recognition(self):
        self.ui.pb_startCamRec.setEnabled(False)
        self.ui.pb_stopFace.setEnabled(True)

        self.ui.label_camera.clear()
        # 置信度
        confidence_default = 0.5
        # 从磁盘加载序列化面部检测器
        proto_path = os.path.sep.join([self.detector_path, "deploy.prototxt"])
        model_path = os.path.sep.join([self.detector_path, "res10_300x300_ssd_iter_140000.caffemodel"])
        detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        # 从磁盘加载序列化面嵌入模型
        try:
            self.ui.textBrowser.append("[INFO] Loading face recognizer...")
            embedded = cv2.dnn.readNetFromTorch(self.embedding_model)
        except IOError:
            self.ui.textBrowser.append("面部嵌入模型的路径不正确！")

        # 加载实际的人脸识别模型和标签
        try:
            recognizer = pickle.loads(open(self.recognizer_path, "rb").read())
            le = pickle.loads(open(self.le_path, "rb").read())
        except IOError:
            self.ui.textBrowser.append("人脸识别模型保存路径不正确！")

        # 循环来自视频文件流的帧
        self.ui.textBrowser.append("Starting Face Recognition...")
        while self.cap.isOpened():
            # 从线程视频流中抓取帧
            ret, frame = self.cap.read()
            QApplication.processEvents()
            if ret:
                # 调整框架的大小以使其宽度为600像素（同时保持纵横比），然后抓取图像尺寸
                frame = imutils.resize(frame, width=600)
                (h, w) = frame.shape[:2]
                # 从图像构造一个blob
                image_blob = cv2.dnn.blobFromImage(
                    cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                    (104.0, 177.0, 123.0), swapRB=False, crop=False)
                # 应用OpenCV的基于深度学习的人脸检测器来定位输入图像中的人脸
                detector.setInput(image_blob)
                detections = detector.forward()
                # 保存识别到的人脸
                face_names = []
                # 循环检测
                for i in range(0, detections.shape[2]):
                    # 提取与预测相关的置信度（即概率）
                    confidence = detections[0, 0, i, 2]

                    # 过滤弱检测
                    if confidence > confidence_default:
                        # 计算面部边界框的（x，y）坐标
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        # 提取面部ROI
                        face = frame[startY:endY, startX:endX]
                        (fH, fW) = face.shape[:2]

                        # 确保面部宽度和高度足够大
                        if fW < 20 or fH < 20:
                            continue

                        # 为面部ROI构造一个blob，然后通过我们的面部嵌入模型传递blob以获得面部的128-d量化
                        face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0),
                                                          swapRB=True,
                                                          crop=False)
                        embedded.setInput(face_blob)
                        vec = embedded.forward()
                        # 执行分类识别面部
                        predicts = recognizer.predict_proba(vec)[0]
                        j = np.argmax(predicts)
                        probability = predicts[j]
                        name = le.classes_[j]
                        # 绘制面部的边界框以及相关的概率
                        text = "{}: {:.2f}%".format(name, probability * 100)
                        y = startY - 10 if startY - 10 > 10 else startY + 10
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                        frame = cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                            (0, 0, 255), 2)
                        face_names.append(name)

                show_video = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 这里指的是显示原图
                # opencv读取图片的样式，不能通过Qlabel进行显示，需要转换为Qimage。
                # QImage(uchar * data, int width, int height, int bytesPerLine, Format format)
                show_image = QImage(show_video.data, show_video.shape[1], show_video.shape[0],
                                    QImage.Format_RGB888)
                self.ui.label_camera.setPixmap(QPixmap.fromImage(show_image))

                # 判断关闭事件是否触发
                if self.th_face_reg_close.is_set():
                    self.ui.textBrowser.append("[INFO] Killing face recognition!")
                    self.ui.textBrowser.moveCursor(self.cursor.End)
                    # 关闭事件为触发，清空显示label
                    self.th_face_reg_close.clear()
                    self.ui.pb_stopFace.setEnabled(False)
                    self.ui.pb_startCamRec.setEnabled(True)
                    self.cap.release()
                    break

        # 因为最后一张画面会显示在GUI中，此处实现清除。
        self.ui.label_camera.clear()


class CollectData(QWidget):
    def __init__(self):
        # super()构造器方法返回父级的对象。__init__()方法是构造器的一个方法。
        super().__init__()
        self.dialog = Ui_Form()
        self.dialog.setupUi(self)

        # 设置窗口名称和图标
        self.setWindowTitle('个人信息采集')
        self.setWindowIcon(QIcon('./imgs/xdu.jpg'))
        # 初始化背景
        self.bkg_pixmap = QPixmap('bkg.png')
        # 设置单张图片背景
        self.dialog.label_capture.setPixmap(self.bkg_pixmap)
        # 导入人脸检测模型
        self.cascade = './haar/haarcascade_frontalface_default.xml'

        # 图片采集保存路径
        self.filepath = "./face_dataset/"

        # 展示图片缩略图的图片类型
        self.img_type = 'png'#'jpg'

        # 设置图片的预览尺寸
        self.display_img_size = 100
        self.col = 0
        self.row = 0
        self.width = 960
        self.height = 600

        # 设置信息采集按键连接函数
        self.dialog.pb_collectInfo.clicked.connect(self.open_cam)
        # 设置拍照按键连接函数
        self.dialog.pb_takePhoto.clicked.connect(self.take_photo)
        # 设置查询信息按键连接函数
        self.dialog.pb_checkInfo.clicked.connect(self.check_info)

        # 初始化信息导入列表
        self.users = []
        # 初始化摄像头
        self.cam_id = 1
        self.cap = cv2.VideoCapture(self.cam_id)
        self.photos = 0

        # 要采集的人脸图像数量
        self.imgs_num = 100

    def handle_click(self):
        if not self.isVisible():
            self.show()

    def handle_close(self):
        self.close()

    def open_cam(self):
        # 判断摄像头是否打开，如果打开则为true，反之为false
        cam_open_flag = self.cap.isOpened()
        if not cam_open_flag:
            # 通过对话框设置被采集人学号
            self.text_name, ok = QInputDialog.getText(self, '创建人脸数据库', '请输入姓名(英文):')
            self.imgs_num,  ok = QInputDialog.getText(self, '创建人脸数据库', '保存图片数量(整数):')
            if ok and self.text_name != '':
                if not self.imgs_num:
                    self.imgs_num = 50
                self.dialog.label_capture.clear()
                self.cap.open(self.cam_id)

                # 启动新的进程，使用display() 函数逐帧读取视频流
                self.th_show_camera = threading.Thread(target=self.show_capture)
                self.th_show_camera.start()

        elif cam_open_flag:
            self.cap.release()
            self.dialog.label_capture.clear()
            self.dialog.pb_collectInfo.setText(u'开始采集')
            # self.dialog.lcdNumber.display(0)

    def show_capture(self):
        self.dialog.pb_collectInfo.setText(u'停止采集')
        self.dialog.label_capture.clear()
        # 加载 Haar级联人脸检测库
        detector = cv2.CascadeClassifier(self.cascade)
        print("[INFO] starting video stream...")
        # 循环来自视频文件流的帧
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            QApplication.processEvents()
            frame = imutils.resize(frame, width=500)
            rects = detector.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1,
                                              minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in rects:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                frame = cv2.putText(frame, "Have token {}/{} faces".format(self.photos, self.imgs_num), (50, 60),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                     (200, 100, 50), 2)
            # 显示输出框架
            show_video = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 这里指的是显示原图
            # opencv读取图片的样式，不能通过Qlabel进行显示，需要转换为Qimage。
            # QImage(uchar * data, int width, int height, int bytesPerLine, Format format)
            self.show_image = QImage(show_video.data, show_video.shape[1], show_video.shape[0], QImage.Format_RGB888)
            self.dialog.label_capture.setPixmap(QPixmap.fromImage(self.show_image))

        QApplication.processEvents()
        # 因为最后一张画面会显示在GUI中，此处实现清除。
        self.dialog.label_capture.clear()
        self.cap.release()

    # 创建文件夹
    # 静态方法可以实现无需实例化调用
    @staticmethod
    def mkdir(path):
        # 去除首位空格
        path = path.strip()
        # 去除尾部 \ 符号
        path = path.rstrip("\\")
        # 判断路径是否存在, 存在=True; 不存在=False
        is_path_exists = os.path.exists(path)
        # 判断结果
        if not is_path_exists:
            # 如果不存在则创建目录
            os.makedirs(path)
            return True

    def take_photo(self):
        if self.cap.isOpened():
            self.photos += 1
            filename = self.filepath + self.text_name + "/"
            self.mkdir(filename)
            photo_save_path = os.path.join(os.path.dirname(os.path.abspath('__file__')), '{}'.format(filename))
            self.show_image.save(photo_save_path + datetime.now().strftime("%Y%m%d%H%M%S") + ".png")
            # p = os.path.sep.join([output, "{}.png".format(str(total).zfill(5))])
            # cv2.imwrite(p, self.show_image)
            self.dialog.lcdNumber.display(self.photos)
            if self.photos == self.imgs_num:
                QMessageBox.information(self, "Information", self.tr("采集成功!"), QMessageBox.Yes | QMessageBox.No)
        else:
            QMessageBox.information(self, '提示', '请先打开摄像头！')

    # 查看一个人所有人脸图片缩略图
    def check_info(self):
        self.dialog.scrollAreaWidgetContents.clearMask()
        file_path = QFileDialog.getExistingDirectory(self, '选择文件夹：', '/')
        if not file_path:
            QMessageBox.information(self, '提示', '文件为空，请重新选择！')
        else:
            print('文件路径：{}'.format(file_path))
            if file_path and self.img_type:
                png_list = list(i for i in os.listdir(file_path) if str(i).endswith('.{}'.format(self.img_type)))
                print("图片列表：", png_list)
                num = len(png_list)
                if num != 0:
                    for i in range(num):
                        # 获取图片完整路径
                        image_path = str(file_path + '/' + png_list[i])
                        pixmap = QPixmap(image_path)
                        self.add_image(pixmap, image_path)
                        QApplication.processEvents()
                else:
                    QMessageBox.warning(self, '错误', '指定路径中不包含{}格式图片！'.format(self.img_type))
                    # self.event(exit())
            else:
                QMessageBox.warning(self, '错误', '未选择文件夹路径！')

    def add_image(self, pixmap, image_path):
        # 图像行数
        n_rows = self.get_image_rows()
        # 这个布局内的数量
        n_widgets = self.dialog.gridLayout.count()
        self.max_rows = n_rows
        if self.row < self.max_rows:
            self.row = self.row + 1
        else:
            self.row = 0
            self.col += 1

        print('行数:{}'.format(self.row), '列数:{}'.format(self.col), '布局内含有的元素数:{}'.format(n_widgets + 1))
        self.dialog.lcdNumber_2.display(n_widgets + 1)
        clickable_image = QClickableImage(self.display_img_size, self.display_img_size, pixmap, image_path)
        self.dialog.gridLayout.addWidget(clickable_image, self.row, self.col)

    def get_image_rows(self):
        # 展示图片的区域
        scroll_area_img_height = self.height
        if scroll_area_img_height > self.display_img_size:
            pic_of_rows = scroll_area_img_height // self.display_img_size  # 计算出一列多少行；
        else:
            pic_of_rows = 1
        return pic_of_rows


class QClickableImage(QWidget):
    image_id = ''

    def __init__(self, width=0, height=0, pixmap=None, image_path=''):
        QWidget.__init__(self)

        self.layout = QVBoxLayout(self)
        self.label1 = QLabel()
        self.label1.setObjectName('label1')
        self.label2 = QLabel()
        self.label2.setObjectName('label2')
        self.width = width
        self.height = height
        self.pixmap = pixmap

        self.label1.clear()
        self.label2.clear()
        if self.width and self.height:
            self.resize(self.width, self.height)
        if self.pixmap:
            pixmap = self.pixmap.scaled(QSize(self.width, self.height),
                                        Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label1.setPixmap(pixmap)
            self.label1.setAlignment(Qt.AlignCenter)
            self.layout.addWidget(self.label1)
        if image_path:
            self.image_id = image_path
            self.label2.setText(image_path)
            self.label2.setAlignment(Qt.AlignCenter)
            # 让文字自适应大小
            self.label2.adjustSize()
            self.layout.addWidget(self.label2)
        self.setLayout(self.layout)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # 自定义主窗口类
    main_wd = MainWindow()
    info_wd = CollectData()
    main_wd.ui.pb_collectFaces.clicked.connect(info_wd.handle_click)
    main_wd.show()

    sys.exit(app.exec())
