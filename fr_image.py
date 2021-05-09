import cv2
import os
import pickle
import imutils
import numpy as np
import time


# IU - Concert Live Clip 2018 Tour.mp4
image_path = "D:/Github/FR-AttSys/test_imgs/IU-2.png"

detector_path = "./face_detection_model"
# OpenCV深度学习面部嵌入模型的路径
embedding_model = "./face_detection_model/openface_nn4.small2.v1.t7"
# 训练模型以识别面部的路径
recognizer_path = "./saved_weights/recognizer.pickle"
# 标签编码器的路径
le_path = "./saved_weights/le.pickle"

faces = ["iu", "pch", "chopin"]
COLORS = np.random.uniform(0, 255, size=(len(faces), 3))


def face_recognition(image_path):
    try:
        image = cv2.imread(image_path)
    except FileExistsError as e:
        print(e)

    # 置信度
    confidence_default = 0.5
    # 从磁盘加载序列化面部检测器
    proto_path = os.path.sep.join([detector_path, "deploy.prototxt"])
    model_path = os.path.sep.join([detector_path, "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)
    # 从磁盘加载序列化面嵌入模型
    try:
        embedded = cv2.dnn.readNetFromTorch(embedding_model)
    except IOError:
        print("面部嵌入模型的路径不正确！")

    # 加载实际的人脸识别模型和标签
    try:
        recognizer = pickle.loads(open(recognizer_path, "rb").read())
        le = pickle.loads(open(le_path, "rb").read())
    except IOError:
        print("人脸识别模型保存路径不正确！")

    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # construct a blob from the image
    image_blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),
                                      (104.0, 177.0, 123.0), swapRB=False, crop=False)

    detector.setInput(image_blob)
    detections = detector.forward()

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
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # 确保面部宽度和高度足够大
            if fW < 20 or fH < 20:
                continue

            # 为面部ROI构造一个blob，然后通过我们的面部嵌入模型传递blob以获得面部的128-d量化
            face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
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

            cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[j], 1)
            image = cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, COLORS[j], 1)

    cv2.imshow("Face Recognition on Image", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    face_recognition(image_path)
