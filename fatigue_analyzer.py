# -*- coding: utf-8 -*-

import time
import config
import speech_utils

class FatigueAnalyzer:
    """疲劳分析模块，负责检测和评估驾驶员疲劳状态"""
    
    def __init__(self, ui_callback=None):
        """初始化疲劳分析器
        
        Args:
            ui_callback: 用于更新UI的回调函数
        """
        self.ui_callback = ui_callback
        
        # 初始化阈值参数
        self.init_thresholds()
        
        # 初始化计数器
        self.init_counters()
    
    def init_thresholds(self):
        """初始化检测阈值"""
        # 眼睛长宽比阈值
        self.EYE_AR_THRESH = config.EYE_AR_THRESH
        self.EYE_AR_CONSEC_FRAMES = config.EYE_AR_CONSEC_FRAMES
        
        # 打哈欠长宽比阈值
        self.MAR_THRESH = config.MAR_THRESH
        self.MOUTH_AR_CONSEC_FRAMES = config.MOUTH_AR_CONSEC_FRAMES
        
        # 瞌睡点头阈值
        self.HAR_THRESH = config.HAR_THRESH
        self.NOD_AR_CONSEC_FRAMES = config.NOD_AR_CONSEC_FRAMES
        
        # 无人驾驶检测阈值
        self.AR_CONSEC_FRAMES_check = config.AR_CONSEC_FRAMES_CHECK
        self.OUT_AR_CONSEC_FRAMES_check = config.OUT_AR_CONSEC_FRAMES_CHECK
    
    def init_counters(self):
        """初始化计数器"""
        # 眨眼相关计数器
        self.COUNTER = 0  # 眨眼帧计数器
        self.TOTAL = 0    # 眨眼总数
        
        # 打哈欠相关计数器
        self.mCOUNTER = 0  # 打哈欠帧计数器
        self.mTOTAL = 0    # 打哈欠总数
        
        # 点头相关计数器
        self.hCOUNTER = 0  # 点头帧计数器
        self.hTOTAL = 0    # 点头总数
        
        # 无人驾驶计数器
        self.oCOUNTER = 0
        
        # 频率计算
        self.frequency = 0   # 眨眼频率
        self.hfrequency = 0  # 点头频率
        self.yfrequency = 0  # 打哈欠频率
        
        # 疲劳评分
        self.score = 0
    
    def update_blink(self, ear):
        """更新眨眼检测
        
        Args:
            ear: 眼睛纵横比
            
        Returns:
            bool: 是否检测到眨眼
        """
        blinked = False
        
        if ear < self.EYE_AR_THRESH:  # 眼睛长宽比小于阈值
            self.COUNTER += 1
        else:
            # 如果连续多帧都小于阈值，则表示进行了一次眨眼活动
            if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                self.TOTAL += 1
                blinked = True
                if self.ui_callback:
                    self.ui_callback(time.strftime('%Y-%m-%d %H:%M ', time.localtime()) + u"眨眼\n")
            # 重置眼帧计数器
            self.COUNTER = 0
        
        return blinked
    
    def update_yawn(self, mar):
        """更新打哈欠检测
        
        Args:
            mar: 嘴部纵横比
            
        Returns:
            bool: 是否检测到打哈欠
        """
        yawned = False
        
        if mar > self.MAR_THRESH:  # 嘴部长宽比大于阈值
            self.mCOUNTER += 1
        else:
            # 如果连续多帧都大于阈值，则表示打了一次哈欠
            if self.mCOUNTER >= self.MOUTH_AR_CONSEC_FRAMES:
                self.mTOTAL += 1
                yawned = True
                if self.ui_callback:
                    self.ui_callback(time.strftime('%Y-%m-%d %H:%M ', time.localtime()) + u"打哈欠\n")
            # 重置嘴帧计数器
            self.mCOUNTER = 0
        
        return yawned
    
    def update_nod(self, euler_angle):
        """更新点头检测
        
        Args:
            euler_angle: 欧拉角
            
        Returns:
            bool: 是否检测到点头
        """
        nodded = False
        
        har = euler_angle[0, 0]  # 取pitch旋转角度
        if har > self.HAR_THRESH:  # 点头角度大于阈值
            self.hCOUNTER += 1
        else:
            # 如果连续多帧都大于阈值，则表示瞌睡点头一次
            if self.hCOUNTER >= self.NOD_AR_CONSEC_FRAMES:
                self.hTOTAL += 1
                nodded = True
                if self.ui_callback:
                    self.ui_callback(time.strftime('%Y-%m-%d %H:%M ', time.localtime()) + u"瞌睡点头\n")
            # 重置点头帧计数器
            self.hCOUNTER = 0
        
        return nodded
    
    def update_no_face(self):
        """更新无人脸检测
        
        Returns:
            bool: 是否检测到无人驾驶状态
        """
        no_driver = False
        
        self.oCOUNTER += 1
        if self.oCOUNTER >= self.OUT_AR_CONSEC_FRAMES_check:
            no_driver = True
            if self.ui_callback:
                self.ui_callback(time.strftime('%Y-%m-%d %H:%M ', time.localtime()) + u"无人驾驶状态!!!\n")
            self.oCOUNTER = 0
        
        return no_driver
    
    def calculate_frequencies(self):
        """计算眨眼、点头和打哈欠的频率"""
        # 记录初始值
        fTOTAL = self.TOTAL
        fmTOTAL = self.mTOTAL
        fhTOTAL = self.hTOTAL
        
        # 等待5秒
        time.sleep(5)
        
        # 记录5秒后的值
        lTOTAL = self.TOTAL
        lmTOTAL = self.mTOTAL
        lhTOTAL = self.hTOTAL
        
        # 计算频率（次数/秒）
        self.frequency = (lTOTAL - fTOTAL) / 5    # 眨眼频率
        self.hfrequency = (lhTOTAL - fhTOTAL) / 5  # 点头频率
        self.yfrequency = (lmTOTAL - fmTOTAL) / 5  # 打哈欠频率
    
    def update_fatigue_score(self):
        """更新疲劳评分"""
        # 确保分数在0-100之间
        if self.score >= 100:
            self.score = 100
        if self.score <= 0:
            self.score = 0
            
        # 根据眨眼频率更新分数
        if self.frequency > 0.47 and self.frequency < 0.61:
            self.score = self.score + 10
        elif self.frequency > 0.62 and self.frequency < 0.95:
            self.score = self.score + 15
        elif self.frequency > 0.96:
            self.score = self.score + 20
        elif self.frequency < 0.47 and self.score >= 0:
            self.score = self.score - 5
            
        # 根据打哈欠频率更新分数
        if self.yfrequency >= 0.2 and self.yfrequency <= 0.4:
            self.score = self.score + 10
        elif self.yfrequency > 0.4 and self.yfrequency <= 0.6:
            self.score = self.score + 15
        elif self.yfrequency > 0.6:
            self.score = self.score + 20
        elif self.yfrequency < 0.2 and self.score >= 0:
            self.score = self.score - 10
            
        # 根据点头频率更新分数
        if self.hfrequency >= 0.2 and self.hfrequency <= 0.4:
            self.score = self.score + 15
        elif self.hfrequency > 0.4 and self.hfrequency <= 0.6:
            self.score = self.score + 20
        elif self.hfrequency > 0.6:
            self.score = self.score + 25
        elif self.hfrequency < 0.2 and self.score >= 0:
            self.score = self.score - 20
            
        # 确保分数在0-100之间
        if self.score >= 100:
            self.score = 100
        if self.score <= 0:
            self.score = 0
    
    def get_fatigue_level(self):
        """获取疲劳等级
        
        Returns:
            str: 疲劳等级 ("normal", "mild", "moderate", "severe")
        """
        if self.score < 30:
            return "normal"
        elif self.score >= 30 and self.score <= 55:
            return "mild"
        elif self.score > 55 and self.score <= 75:
            return "moderate"
        else:  # self.score > 75
            return "severe"
    
    def check_and_alert(self):
        """检查疲劳状态并发出警报"""
        fatigue_level = self.get_fatigue_level()
        
        if fatigue_level == "mild":
            if self.ui_callback:
                self.ui_callback(time.strftime('%Y-%m-%d %H:%M ', time.localtime()) + 
                               u"警报警报已进入轻度疲劳，请打起精神！！\n准备开始语音播报\n")
            speech_utils.speak(speech_utils.MESSAGES["fatigue_mild"])
        
        elif fatigue_level == "moderate":
            if self.ui_callback:
                self.ui_callback(time.strftime('%Y-%m-%d %H:%M ', time.localtime()) + 
                               u"警报警报已进入中度疲劳，请尽快打起精神！！！\n准备开始语音播报\n")
            speech_utils.speak(speech_utils.MESSAGES["fatigue_moderate"])
        
        elif fatigue_level == "severe":
            if self.ui_callback:
                self.ui_callback(time.strftime('%Y-%m-%d %H:%M ', time.localtime()) + 
                               u"警报警报已进入重度疲劳，请靠边停车，已为您自动报警\n准备开始语音播报\n")
            speech_utils.speak(speech_utils.MESSAGES["fatigue_severe"])
    
    def get_status_info(self):
        """获取当前状态信息
        
        Returns:
            dict: 包含当前状态的字典
        """
        return {
            "blinks": self.TOTAL,
            "yawns": self.mTOTAL,
            "nods": self.hTOTAL,
            "blink_frequency": self.frequency,
            "yawn_frequency": self.yfrequency,
            "nod_frequency": self.hfrequency,
            "fatigue_score": self.score,
            "fatigue_level": self.get_fatigue_level()
        }