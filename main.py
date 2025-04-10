# -*- coding: utf-8 -*-

import wx
import face_detector
import fatigue_analyzer
import ui_components

class FatigueDetectionApp(wx.App):
    """疲劳驾驶检测应用程序"""
    
    def OnInit(self):
        """初始化应用程序"""
        # 创建人脸检测器
        detector = face_detector.FaceDetector()
        
        # 创建疲劳分析器
        analyzer = fatigue_analyzer.FatigueAnalyzer()
        
        # 创建UI界面
        self.frame = ui_components.FatigueDetectionUI(parent=None, title="疲劳驾驶检测", detector=detector, analyzer=analyzer)
        self.frame.Show(True)
        return True

if __name__ == "__main__":
    app = FatigueDetectionApp()
    app.MainLoop()