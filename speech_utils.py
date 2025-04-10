# -*- coding: utf-8 -*-

import pyttsx3
import pythoncom
from win32com import client

def speak_windows(text):
    """
    使用Windows SAPI接口进行语音播报
    
    Args:
        text: 要播报的文本内容
    """
    try:
        pythoncom.CoInitialize()
        engine = client.Dispatch("SAPI.SpVoice")
        engine.Speak(text)
        return True
    except Exception as e:
        print(f"Windows语音播报失败: {e}")
        return False

def speak_pyttsx3(text):
    """
    使用pyttsx3库进行语音播报（作为备选方案）
    
    Args:
        text: 要播报的文本内容
    """
    try:
        engine = pyttsx3.init()
        print('准备开始语音播报...')
        engine.say(text)
        engine.runAndWait()
        return True
    except Exception as e:
        print(f"pyttsx3语音播报失败: {e}")
        return False

def speak(text, use_windows=True):
    """
    统一的语音播报接口
    
    Args:
        text: 要播报的文本内容
        use_windows: 是否优先使用Windows SAPI接口
    
    Returns:
        bool: 播报是否成功
    """
    if use_windows:
        if speak_windows(text):
            return True
        # 如果Windows SAPI失败，尝试使用pyttsx3
        return speak_pyttsx3(text)
    else:
        return speak_pyttsx3(text)

# 预定义的语音消息
MESSAGES = {
    "camera_success": "打开摄像头成功，开始为您检测，祝您一路顺风",
    "camera_fail": "打开摄像头失败，请重试",
    "model_loaded": "初始化车载摄像头成功",
    "exit": "再见，欢迎您再次使用，祝您一路平安",
    "fatigue_mild": "警报警报，检测到您已进入轻度疲劳，请注意",
    "fatigue_moderate": "警报警报，检测到您已进入中度疲劳，请尽快打起精神，否则即将自动报警",
    "fatigue_severe": "警报警报，检测到您已进入重度疲劳，请靠边停车，已为您自动报警"
}