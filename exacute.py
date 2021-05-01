# -*- coding:utf-8 -*-
import os
import sys
import threading
from functools import wraps
import time
import cv2
import imutils
from PyQt5 import QtWidgets
from PyQt5.QtCore import QCoreApplication, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QIcon, QImage
from PyQt5.QtWidgets import QApplication, QProgressBar, QStyle
from PyQt5.QtWidgets import QFileDialog, QMessageBox

# 导入自定义包
from ui.WindowUI import Ui_MainWindow


# progress bar class
class ProBar(QThread):
    bar_signal = pyqtSignal(int)
    def __int__(self):
        super(ProBar, self).__init__()

    def run(self):
        for i in range(100+1):
            # time.sleep(0.05)
            self.bar_signal.emit(i)


class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        # 继承ui窗口类
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.retranslateUi(self)

        # ####################### 窗口初始化 ######################
        # 设置图片背景
        self.ui.label_camera.setPixmap(QPixmap('./imgs/bkg.png'))
        # 设置窗口名称和图标
        self.setWindowTitle("FR-Sys")
        self.setWindowIcon(QIcon('./imgs/xdu.jpg'))

        # # QTimer导致卡顿
        # # 设置显示日期和时间
        # timer = QTimer(self)
        # timer.timeout.connect(self.show_time_text)
        # timer.start()

        # ####################### 定义标识符 ######################
        try:
            print("Starting initialize Camera···")
            # 初始化摄像头
            self.cap = cv2.VideoCapture(cv2.CAP_DSHOW + 1)
            # 设置显示分辨率和FPS，否则很卡
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
            self.cap.set(cv2.CAP_PROP_FPS, 60)

        except EnvironmentError as e:
            print("Failed to initialized Camera!", e)
        finally:
            print("Finished initialize Camera!")

        # 定义每个人需要采集的人脸数
        self.num_photos_of_person = 30
        # 定义打开相机标识符
        self.flag_open_camera = False
        # 设置相机打开和关闭按键的初始状态
        self.ui.pb_openCamera.setEnabled(True)
        self.ui.pb_closeCamera.setEnabled(False)
        # 定义打开文件类型标识符
        self.flag_open_file_type = None
        # 获取当前文件路径
        self.cwd = os.getcwd()

        # progress bar
        # self.ui.progressBar.setGeometry(0, 0, 300, 25)
        self.ui.progressBar.setValue(0)
        # self.ui.progressBar.setMaximum(100)

        # #################### 定义按键连接函数 ####################
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

    # ############################################################
    # # 阻塞！！！！！！
    # # 显示系统时间以及相关文字提示函数
    # def show_time_text(self):
    #     # 设置宽度
    #     self.ui.label_time.setFixedWidth(200)
    #     # 设置显示文本格式
    #     self.ui.label_time.setStyleSheet(
    #         # "QLabel{background:white;}" 此处设置背景色
    #         # "QLabel{color:rgb(300,300,300,120); font-size:14px; font-weight:bold; font-family:宋体;}"
    #         "QLabel{font-size:14px; font-weight:bold; font-family:宋体;}")
    #     datetime = QDateTime.currentDateTime().toString()
    #     self.ui.label_time.setText("" + datetime)

    def pb_close_camera(self):
        """
        关闭相机按键的槽函数，用于触发关闭摄像头线程Event，以便关闭摄像头
        :return:
        """
        print("[INFO] killing Camera stream...")
        # 触发关闭摄像头线程事件
        self.th_camera_close.set()

    def pb_close_file(self):
        """
        设置关闭视频文件的按键连接函数
        :return:
        """
        self.th_video_close.set()

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
            return call_in_func
        return wrap_func

    @button_useful
    def rb_open_image(self,  *args, **kwargs):
        self.flag_open_file_type = "image"

    @button_useful
    def rb_open_video(self,  *args, **kwargs):
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
                        print("Cancel Image Select!")
                    else:
                        print(self.file_name[0])
                        # 一次识别一张图片
                        self.ui.label_videoFile.setPixmap(QPixmap(self.file_name[0]))
                else:
                    self.file_name, self.file_type = QFileDialog.getOpenFileName(self, 'Choose file', self.cwd, '*.mp4')

                    # 如果未选择文件，则不执行打开操作
                    if len(self.file_name) == 0:
                        print("Cancel Video Select!")
                    else:
                        self.cap2 = cv2.VideoCapture(self.file_name)
                        self.fps2 = self.cap2.get(cv2.CAP_PROP_FPS)

                        # 设置显示分辨率和FPS，否则很卡
                        self.cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
                        self.cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)
                        self.cap2.set(cv2.CAP_PROP_FPS, 60)

                        # 启动新的进程，使用display() 函数逐帧读取视频流
                        self.th_display_video = threading.Thread(target=self.display_video)
                        self.th_display_video.start()

        # 打开摄像头标志位为True，调用display获取主机视频流
        else:
            # 创建一个打开摄像头的线程
            self.th_camera_open = threading.Thread(target=self.display_camera, daemon=True)
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
                    print("Starting close Video···")
                    # 关闭事件为触发，清空显示label
                    self.th_video_close.clear()
                    self.ui.label_videoFile.clear()
                    self.ui.lcd_fps1.display(0)
                    self.ui.progressBar.setValue(0)

                    print("Succeed to close Video")
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
        self.rb_group_checked(False)
        self.ui.label_camera.clear()

        # 导入opencv人脸检测xml文件
        cascade = './haar/haarcascade_frontalface_default.xml'
        # 加载 Haar级联人脸检测库
        detector = cv2.CascadeClassifier(cascade)
        print("[INFO] starting Camera stream...")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        # 循环来自视频文件流的帧
        while self.cap.isOpened():
            success, frame = self.cap.read()
            QApplication.processEvents()
            frame2 = imutils.resize(frame, width=600)  # self.ui.label_camera.width()
            rects = detector.detectMultiScale(cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), scaleFactor=1.1,
                                              minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in rects:
                cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)
                frame2 = cv2.putText(frame2, "Detecting Faces",
                                     (50, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                     0.7, (200, 100, 50), 2)
            # 显示输出框架
            # 这里指的是显示原图
            show_video2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            # opencv读取图片的样式，不能通过Qlabel进行显示，需要转换为Qimage。
            # QImage(uchar * data, int width, int height, int bytesPerLine, Format format)
            video_img = QImage(show_video2.data, show_video2.shape[1], show_video2.shape[0], QImage.Format_RGB888)
            self.ui.label_camera.setPixmap(QPixmap.fromImage(video_img))

            # 显示FPS
            self.ui.lcd_fps2.display("%d" % self.fps)
            # 判断摄像头关闭事件是否触发
            if self.th_camera_close.is_set():
                # 关闭事件为触发，清空显示label
                self.th_camera_close.clear()
                # 恢复用于显示摄像头内容的区域的背景
                self.ui.label_camera.setPixmap(QPixmap('./imgs/bkg.png'))
                self.ui.pb_closeCamera.setEnabled(False)
                self.ui.pb_openCamera.setEnabled(True)
                break

        # 因为最后一张画面会显示在GUI中，此处实现清除。
        self.ui.label_camera.clear()
        self.ui.label_camera.setPixmap(QPixmap('./imgs/bkg.png'))
        self.ui.lcd_fps2.display(0)
        self.rb_group_checked(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # 自定义主窗口类
    wd = MyWindow()
    wd.show()

    sys.exit(app.exec())
