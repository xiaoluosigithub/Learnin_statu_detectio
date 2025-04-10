# -*- coding: utf-8 -*-

import dlib
import cv2
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import math

class FaceDetector:
    """人脸检测和特征提取模块"""
    
    def __init__(self, predictor_path="./model/shape_predictor_68_face_landmarks.dat"):
        # 使用人脸检测器get_frontal_face_detector
        self.detector = dlib.get_frontal_face_detector()
        # dlib的68点模型，使用作者训练好的特征预测器
        self.predictor = dlib.shape_predictor(predictor_path)
        
        # 分别获取左右眼面部标志的索引
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (self.mStart, self.mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        
        # 姿态估计相关参数
        self.init_pose_params()
    
    def init_pose_params(self):
        """初始化姿态估计相关参数"""
        # 从配置文件导入参数
        import config
        
        # 世界坐标系(UVW)：填写3D参考点
        self.object_pts = config.OBJECT_PTS

        # 相机坐标系(XYZ)：添加相机内参
        self.K = config.K  # 等价于矩阵[fx, 0, cx; 0, fy, cy; 0, 0, 1]
        # 图像中心坐标系(uv)：相机畸变参数[k1, k2, p1, p2, k3]
        self.D = config.D

        # 像素坐标系(xy)：填写凸轮的本征和畸变系数
        self.cam_matrix = np.array(self.K).reshape(3, 3).astype(np.float32)
        self.dist_coeffs = np.array(self.D).reshape(5, 1).astype(np.float32)

        # 重新投影3D点的世界坐标轴以验证结果姿势
        self.reprojectsrc = config.REPROJECTSRC
        # 绘制正方体12轴
        self.line_pairs = config.LINE_PAIRS
    
    def get_head_pose(self, shape):
        """头部姿态估计
        
        Args:
            shape: 人脸特征点坐标
            
        Returns:
            tuple: (投影误差，欧拉角)
        """
        # （像素坐标集合）填写2D参考点，注释遵循https://ibug.doc.ic.ac.uk/resources/300-W/
        # 17左眉左上角/21左眉右角/22右眉左上角/26右眉右上角/36左眼左上角/39左眼右上角/42右眼左上角/
        # 45右眼右上角/31鼻子左上角/35鼻子右上角/48左上角/54嘴右上角/57嘴中央下角/8下巴角
        image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                                shape[39], shape[42], shape[45], shape[31], shape[35],
                                shape[48], shape[54], shape[57], shape[8]])
        # solvePnP计算姿势——求解旋转和平移矩阵：
        # rotation_vec表示旋转矩阵，translation_vec表示平移矩阵，cam_matrix与K矩阵对应，dist_coeffs与D矩阵对应。
        _, rotation_vec, translation_vec = cv2.solvePnP(self.object_pts, image_pts, self.cam_matrix, self.dist_coeffs)
        # projectPoints重新投影误差：原2d点和重投影2d点的距离（输入3d点、相机内参、相机畸变、r、t，输出重投影2d点）
        reprojectdst, _ = cv2.projectPoints(self.reprojectsrc, rotation_vec, translation_vec, self.cam_matrix,
                                            self.dist_coeffs)
        reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))  # 以8行2列显示

        # 计算欧拉角calc euler angle
        # 参考https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#decomposeprojectionmatrix
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)  # 罗德里格斯公式（将旋转矩阵转换为旋转向量）
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))  # 水平拼接，vconcat垂直拼接
        # decomposeProjectionMatrix将投影矩阵分解为旋转矩阵和相机矩阵
        _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

        pitch, yaw, roll = [math.radians(_.item()) for _ in euler_angle]

        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))
        # print('pitch:{}, yaw:{}, roll:{}'.format(pitch, yaw, roll))

        return reprojectdst, euler_angle  # 投影误差，欧拉角

    def eye_aspect_ratio(self, eye):
        """计算眼睛纵横比
        
        Args:
            eye: 眼睛特征点坐标
            
        Returns:
            float: 眼睛纵横比
        """
        # 垂直眼标志（X，Y）坐标
        A = dist.euclidean(eye[1], eye[5])  # 计算两个集合之间的欧式距离
        B = dist.euclidean(eye[2], eye[4])
        # 计算水平之间的欧几里得距离
        # 水平眼标志（X，Y）坐标
        C = dist.euclidean(eye[0], eye[3])
        # 眼睛长宽比的计算
        ear = (A + B) / (2.0 * C)
        # 返回眼睛的长宽比
        return ear

    def mouth_aspect_ratio(self, mouth):
        """计算嘴部纵横比
        
        Args:
            mouth: 嘴部特征点坐标
            
        Returns:
            float: 嘴部纵横比
        """
        A = np.linalg.norm(mouth[2] - mouth[9])  # 51, 59
        B = np.linalg.norm(mouth[4] - mouth[7])  # 53, 57
        C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
        mar = (A + B) / (2.0 * C)
        return mar
    
    def detect_face(self, frame):
        """检测人脸并提取特征
        
        Args:
            frame: 视频帧
            
        Returns:
            tuple: (是否检测到人脸, 人脸特征点, 左眼坐标, 右眼坐标, 嘴部坐标, 眼睛纵横比, 嘴部纵横比, 头部姿态)
        """
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 检测人脸
        faces = self.detector(gray, 0)
        
        # 如果没有检测到人脸
        if len(faces) == 0:
            return False, None, None, None, None, None, None, None
        
        # 获取第一个人脸
        face = faces[0]
        
        # 获取人脸特征点
        shape = self.predictor(frame, face)
        shape = face_utils.shape_to_np(shape)
        
        # 获取左眼和右眼坐标
        leftEye = shape[self.lStart:self.lEnd]
        rightEye = shape[self.rStart:self.rEnd]
        
        # 获取嘴部坐标
        mouth = shape[self.mStart:self.mEnd]
        
        # 计算眼睛纵横比
        leftEAR = self.eye_aspect_ratio(leftEye)
        rightEAR = self.eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        
        # 计算嘴部纵横比
        mar = self.mouth_aspect_ratio(mouth)
        
        # 获取头部姿态
        pose = self.get_head_pose(shape)
        
        return True, shape, leftEye, rightEye, mouth, ear, mar, pose
    
    def draw_face_features(self, frame, shape, leftEye, rightEye, mouth):
        """在图像上绘制人脸特征
        
        Args:
            frame: 视频帧
            shape: 人脸特征点
            leftEye: 左眼坐标
            rightEye: 右眼坐标
            mouth: 嘴部坐标
            
        Returns:
            ndarray: 绘制了特征的图像
        """
        # 绘制人脸矩形框
        face = dlib.rectangle(int(shape[0, 0]), int(shape[0, 1]), int(shape[-1, 0]), int(shape[-1, 1]))
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 1)
        
        # 绘制人脸特征点
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1, 8)
        
        # 绘制眼睛轮廓
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        
        # 绘制嘴部轮廓
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        
        return frame