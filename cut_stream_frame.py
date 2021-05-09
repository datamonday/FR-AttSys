import cv2
import numpy as np
from datetime import datetime


dir_name = "IU"
save_path = "D:/Github/FR-AttSys/face_dataset/"
video_path = "D:/Github/FR-AttSys/test_video/[IU] Meaning Of You 190428.mp4"


def cut_video_frame():
    cap = cv2.VideoCapture(video_path)
    # 定时装置初始值
    i = 0
    # 文件名序号初始值
    save_name_id = 1

    while cap.isOpened():
        i = i + 1
        reg, frame = cap.read()
        # 图片左右调换
        frame = cv2.flip(frame, 1)
        # 显示视频
        cv2.imshow('Cut Frame from Stream', frame)

        # 定时截屏
        if i == 50:
            filename = str(save_name_id) + '-' + datetime.now().strftime("%Y%m%d%H%M%S") + '.png'
            # 截图 前面为放在桌面的路径 frame为此时的图像
            cv2.imwrite(save_path + dir_name + '/' + filename, frame)
            print(f"[Info] {filename} saved successfully!")
            # 清零
            i = 0

            save_name_id += 1

            # 最多截图20张
            if save_name_id >= 2:
                break

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    # 释放资源
    cap.release()


if __name__ == "__main__":
    cut_video_frame()
    cv2.destroyAllWindows()