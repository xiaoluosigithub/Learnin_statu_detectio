# -*- coding: utf-8 -*-

import wx
import wx.xrc
import wx.adv
import cv2
import time
import _thread
import numpy as np
import config
import speech_utils

class FatigueDetectionUI(wx.Frame):
    """疲劳驾驶检测系统UI组件"""
    
    def __init__(self, parent, title, detector, analyzer):
        """初始化UI组件
        
        Args:
            parent: 父窗口
            title: 窗口标题
            detector: 人脸检测器实例
            analyzer: 疲劳分析器实例
        """
        wx.Frame.__init__(self, parent, id=wx.ID_ANY, title=title, pos=wx.DefaultPosition, size=wx.Size(925, 535),
                          style=wx.DEFAULT_FRAME_STYLE | wx.TAB_TRAVERSAL)
        
        # 保存检测器和分析器实例
        self.detector = detector
        self.analyzer = analyzer
        
        # 设置分析器的UI回调函数
        self.analyzer.ui_callback = self.append_text
        
        # 初始化UI组件
        self.init_ui()
        
        # 初始化摄像头参数
        self.init_camera_params()
        
        # 绑定事件处理函数
        self.bind_events()
    
    def init_ui(self):
        """初始化UI组件"""
        self.SetSizeHints(wx.DefaultSize, wx.DefaultSize)
        self.SetBackgroundColour(wx.Colour(240, 240, 245))  # 使用更柔和的背景色

        # 创建主布局
        bSizer1 = wx.BoxSizer(wx.VERTICAL)
        
        # 添加标题面板
        title_panel = wx.Panel(self, wx.ID_ANY, wx.DefaultPosition, wx.Size(-1, 50), wx.TAB_TRAVERSAL)
        title_panel.SetBackgroundColour(wx.Colour(41, 128, 185))  # 蓝色标题栏
        title_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # 添加标题文本
        title_text = wx.StaticText(title_panel, wx.ID_ANY, u"疲劳驾驶检测系统", wx.DefaultPosition, wx.DefaultSize, 0)
        title_text.SetFont(wx.Font(16, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False, ""))
        title_text.SetForegroundColour(wx.Colour(255, 255, 255))
        title_sizer.Add(title_text, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 10)
        
        title_panel.SetSizer(title_sizer)
        bSizer1.Add(title_panel, 0, wx.EXPAND, 5)
        
        # 添加主内容区域
        content_panel = wx.Panel(self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL)
        content_panel.SetBackgroundColour(wx.Colour(240, 240, 245))
        bSizer2 = wx.BoxSizer(wx.HORIZONTAL)
        
        # 创建视频显示区域（左侧）
        video_panel = wx.Panel(content_panel, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.BORDER_SIMPLE)
        video_panel.SetBackgroundColour(wx.Colour(255, 255, 255))
        bSizer3 = wx.BoxSizer(wx.VERTICAL)
        
        # 视频标题
        video_title = wx.StaticText(video_panel, wx.ID_ANY, u"实时监控", wx.DefaultPosition, wx.DefaultSize, 0)
        video_title.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False, ""))
        video_title.SetForegroundColour(wx.Colour(70, 70, 70))
        bSizer3.Add(video_title, 0, wx.ALL, 10)
        
        # 视频控件
        self.m_animCtrl1 = wx.adv.AnimationCtrl(video_panel, wx.ID_ANY, wx.adv.NullAnimation, wx.DefaultPosition,
                                                wx.DefaultSize, wx.adv.AC_DEFAULT_STYLE)
        bSizer3.Add(self.m_animCtrl1, 1, wx.ALL | wx.EXPAND, 10)
        
        video_panel.SetSizer(bSizer3)
        bSizer2.Add(video_panel, 7, wx.EXPAND | wx.ALL, 10)
        
        # 创建控制面板（右侧）
        control_panel = wx.Panel(content_panel, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL)
        control_panel.SetBackgroundColour(wx.Colour(255, 255, 255))
        bSizer4 = wx.BoxSizer(wx.VERTICAL)
        
        # 控制面板标题
        control_title = wx.StaticText(control_panel, wx.ID_ANY, u"控制中心", wx.DefaultPosition, wx.DefaultSize, 0)
        control_title.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False, ""))
        control_title.SetForegroundColour(wx.Colour(70, 70, 70))
        bSizer4.Add(control_title, 0, wx.ALL, 10)
        
        # 创建按钮面板
        button_panel = wx.Panel(control_panel, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL)
        button_panel.SetBackgroundColour(wx.Colour(245, 245, 250))
        sbSizer2 = wx.BoxSizer(wx.VERTICAL)
        
        # 按钮面板标题
        button_title = wx.StaticText(button_panel, wx.ID_ANY, u"操作区", wx.DefaultPosition, wx.DefaultSize, 0)
        button_title.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False, ""))
        sbSizer2.Add(button_title, 0, wx.ALL, 10)
        
        # 创建按钮网格
        gSizer1 = wx.GridSizer(0, 2, 10, 10)
        
        # 加载摄像头按钮 - 蓝色
        self.m_choice1 = wx.Button(button_panel, wx.ID_ANY, u"加载车载摄像头", wx.DefaultPosition, wx.Size(130, 60), 0)
        self.m_choice1.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False, ""))
        self.m_choice1.SetBackgroundColour(wx.Colour(52, 152, 219))
        self.m_choice1.SetForegroundColour(wx.Colour(255, 255, 255))
        gSizer1.Add(self.m_choice1, 0, wx.EXPAND, 5)
        
        # 开始检测按钮 - 绿色
        self.camera_button1 = wx.Button(button_panel, wx.ID_ANY, u"开始检测", wx.DefaultPosition, wx.Size(130, 60), 0)
        self.camera_button1.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False, ""))
        self.camera_button1.SetBackgroundColour(wx.Colour(46, 204, 113))
        self.camera_button1.SetForegroundColour(wx.Colour(255, 255, 255))
        gSizer1.Add(self.camera_button1, 0, wx.EXPAND, 5)
        
        # 暂停按钮 - 橙色
        self.off_button3 = wx.Button(button_panel, wx.ID_ANY, u"暂停", wx.DefaultPosition, wx.Size(130, 60), 0)
        self.off_button3.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False, ""))
        self.off_button3.SetBackgroundColour(wx.Colour(230, 126, 34))
        self.off_button3.SetForegroundColour(wx.Colour(255, 255, 255))
        gSizer1.Add(self.off_button3, 0, wx.EXPAND, 5)
        
        # 退出检测按钮 - 红色
        self.off_button4 = wx.Button(button_panel, wx.ID_ANY, u"退出检测", wx.DefaultPosition, wx.Size(130, 60), 0)
        self.off_button4.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False, ""))
        self.off_button4.SetBackgroundColour(wx.Colour(231, 76, 60))
        self.off_button4.SetForegroundColour(wx.Colour(255, 255, 255))
        gSizer1.Add(self.off_button4, 0, wx.EXPAND, 5)
        
        # 添加按钮网格到布局
        sbSizer2.Add(gSizer1, 0, wx.ALL | wx.EXPAND, 10)
        button_panel.SetSizer(sbSizer2)
        bSizer4.Add(button_panel, 0, wx.ALL | wx.EXPAND, 10)
        
        # 创建疲劳状态指示面板
        fatigue_panel = wx.Panel(control_panel, wx.ID_ANY, wx.DefaultPosition, wx.Size(-1, 100), wx.TAB_TRAVERSAL)
        fatigue_panel.SetBackgroundColour(wx.Colour(245, 245, 250))
        fatigue_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # 疲劳状态面板标题
        fatigue_title = wx.StaticText(fatigue_panel, wx.ID_ANY, u"疲劳状态", wx.DefaultPosition, wx.DefaultSize, 0)
        fatigue_title.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False, ""))
        fatigue_sizer.Add(fatigue_title, 0, wx.ALL, 10)
        
        # 疲劳状态指示器
        indicator_panel = wx.Panel(fatigue_panel, wx.ID_ANY, wx.DefaultPosition, wx.Size(-1, 30), wx.TAB_TRAVERSAL)
        indicator_panel.SetBackgroundColour(wx.Colour(255, 255, 255))
        indicator_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # 四个状态指示灯
        self.normal_indicator = wx.Panel(indicator_panel, wx.ID_ANY, wx.DefaultPosition, wx.Size(50, 20), wx.BORDER_SIMPLE)
        self.normal_indicator.SetBackgroundColour(wx.Colour(46, 204, 113))
        self.normal_text = wx.StaticText(indicator_panel, wx.ID_ANY, u"正常", wx.DefaultPosition, wx.DefaultSize, 0)
        
        self.mild_indicator = wx.Panel(indicator_panel, wx.ID_ANY, wx.DefaultPosition, wx.Size(50, 20), wx.BORDER_SIMPLE)
        self.mild_indicator.SetBackgroundColour(wx.Colour(230, 126, 34))
        self.mild_indicator.SetBackgroundColour(wx.Colour(200, 200, 200))  # 默认灰色
        self.mild_text = wx.StaticText(indicator_panel, wx.ID_ANY, u"轻度", wx.DefaultPosition, wx.DefaultSize, 0)
        
        self.moderate_indicator = wx.Panel(indicator_panel, wx.ID_ANY, wx.DefaultPosition, wx.Size(50, 20), wx.BORDER_SIMPLE)
        self.moderate_indicator.SetBackgroundColour(wx.Colour(200, 200, 200))  # 默认灰色
        self.moderate_text = wx.StaticText(indicator_panel, wx.ID_ANY, u"中度", wx.DefaultPosition, wx.DefaultSize, 0)
        
        self.severe_indicator = wx.Panel(indicator_panel, wx.ID_ANY, wx.DefaultPosition, wx.Size(50, 20), wx.BORDER_SIMPLE)
        self.severe_indicator.SetBackgroundColour(wx.Colour(200, 200, 200))  # 默认灰色
        self.severe_text = wx.StaticText(indicator_panel, wx.ID_ANY, u"重度", wx.DefaultPosition, wx.DefaultSize, 0)
        
        # 添加指示灯到布局
        indicator_sizer.Add(self.normal_indicator, 0, wx.ALL, 5)
        indicator_sizer.Add(self.normal_text, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        indicator_sizer.Add(self.mild_indicator, 0, wx.ALL, 5)
        indicator_sizer.Add(self.mild_text, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        indicator_sizer.Add(self.moderate_indicator, 0, wx.ALL, 5)
        indicator_sizer.Add(self.moderate_text, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        indicator_sizer.Add(self.severe_indicator, 0, wx.ALL, 5)
        indicator_sizer.Add(self.severe_text, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        
        indicator_panel.SetSizer(indicator_sizer)
        fatigue_sizer.Add(indicator_panel, 0, wx.ALL | wx.EXPAND, 5)
        
        fatigue_panel.SetSizer(fatigue_sizer)
        bSizer4.Add(fatigue_panel, 0, wx.ALL | wx.EXPAND, 10)
        
        # 创建数据统计面板
        stats_panel = wx.Panel(control_panel, wx.ID_ANY, wx.DefaultPosition, wx.Size(-1, 120), wx.TAB_TRAVERSAL)
        stats_panel.SetBackgroundColour(wx.Colour(245, 245, 250))
        stats_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # 数据统计面板标题
        stats_title = wx.StaticText(stats_panel, wx.ID_ANY, u"检测数据", wx.DefaultPosition, wx.DefaultSize, 0)
        stats_title.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False, ""))
        stats_sizer.Add(stats_title, 0, wx.ALL, 10)
        
        # 数据网格
        stats_grid = wx.FlexGridSizer(3, 2, 5, 20)
        
        # 眨眼次数
        blink_label = wx.StaticText(stats_panel, wx.ID_ANY, u"眨眼次数:", wx.DefaultPosition, wx.DefaultSize, 0)
        self.blink_value = wx.StaticText(stats_panel, wx.ID_ANY, u"0", wx.DefaultPosition, wx.DefaultSize, 0)
        self.blink_value.SetFont(wx.Font(9, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False, ""))
        
        # 打哈欠次数
        yawn_label = wx.StaticText(stats_panel, wx.ID_ANY, u"打哈欠次数:", wx.DefaultPosition, wx.DefaultSize, 0)
        self.yawn_value = wx.StaticText(stats_panel, wx.ID_ANY, u"0", wx.DefaultPosition, wx.DefaultSize, 0)
        self.yawn_value.SetFont(wx.Font(9, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False, ""))
        
        # 点头次数
        nod_label = wx.StaticText(stats_panel, wx.ID_ANY, u"点头次数:", wx.DefaultPosition, wx.DefaultSize, 0)
        self.nod_value = wx.StaticText(stats_panel, wx.ID_ANY, u"0", wx.DefaultPosition, wx.DefaultSize, 0)
        self.nod_value.SetFont(wx.Font(9, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False, ""))
        
        # 添加到网格
        stats_grid.Add(blink_label, 0, wx.ALIGN_CENTER_VERTICAL, 5)
        stats_grid.Add(self.blink_value, 0, wx.ALIGN_CENTER_VERTICAL, 5)
        stats_grid.Add(yawn_label, 0, wx.ALIGN_CENTER_VERTICAL, 5)
        stats_grid.Add(self.yawn_value, 0, wx.ALIGN_CENTER_VERTICAL, 5)
        stats_grid.Add(nod_label, 0, wx.ALIGN_CENTER_VERTICAL, 5)
        stats_grid.Add(self.nod_value, 0, wx.ALIGN_CENTER_VERTICAL, 5)
        
        stats_sizer.Add(stats_grid, 0, wx.ALL, 10)
        stats_panel.SetSizer(stats_sizer)
        bSizer4.Add(stats_panel, 0, wx.ALL | wx.EXPAND, 10)
        
        # 创建状态输出区域
        status_panel = wx.Panel(control_panel, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL)
        status_panel.SetBackgroundColour(wx.Colour(245, 245, 250))
        sbSizer6 = wx.BoxSizer(wx.VERTICAL)
        
        # 状态面板标题
        status_title = wx.StaticText(status_panel, wx.ID_ANY, u"状态输出", wx.DefaultPosition, wx.DefaultSize, 0)
        status_title.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False, ""))
        sbSizer6.Add(status_title, 0, wx.ALL, 10)
        
        # 状态文本控件 - 使用更美观的样式
        self.m_textCtrl3 = wx.TextCtrl(status_panel, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition,
                                       wx.DefaultSize, wx.TE_MULTILINE | wx.TE_READONLY | wx.BORDER_THEME)
        self.m_textCtrl3.SetFont(wx.Font(9, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, False, ""))
        self.m_textCtrl3.SetBackgroundColour(wx.Colour(250, 250, 250))
        self.m_textCtrl3.SetForegroundColour(wx.Colour(50, 50, 50))
        
        # 添加初始欢迎信息
        welcome_time = time.strftime('%Y-%m-%d %H:%M ', time.localtime())
        self.m_textCtrl3.AppendText(welcome_time + u"欢迎使用疲劳驾驶检测系统\n")
        self.m_textCtrl3.AppendText(u"请点击'加载车载摄像头'按钮开始使用\n")
        
        sbSizer6.Add(self.m_textCtrl3, 1, wx.ALL | wx.EXPAND, 10)
        
        status_panel.SetSizer(sbSizer6)
        bSizer4.Add(status_panel, 1, wx.ALL | wx.EXPAND, 10)
        
        # 完成控制面板布局
        control_panel.SetSizer(bSizer4)
        bSizer2.Add(control_panel, 3, wx.EXPAND | wx.ALL, 10)
        
        # 完成内容区域布局
        content_panel.SetSizer(bSizer2)
        bSizer1.Add(content_panel, 1, wx.EXPAND, 5)
        
        # 添加状态栏
        status_bar = self.CreateStatusBar(1, wx.STB_SIZEGRIP, wx.ID_ANY)
        status_bar.SetStatusText(u"疲劳驾驶检测系统 - 准备就绪")
        
        # 完成主布局
        self.SetSizer(bSizer1)
        self.Layout()
        self.Centre(wx.BOTH)
        
        # 加载封面图片
        self.image_cover = wx.Image(config.COVER, wx.BITMAP_TYPE_ANY)
        
        # 获取视频控件的大小
        anim_size = self.m_animCtrl1.GetClientSize()
        panel_width, panel_height = anim_size.GetWidth(), anim_size.GetHeight()
        
        # 如果无法获取有效的尺寸，使用默认尺寸
        if panel_width <= 0 or panel_height <= 0:
            panel_width, panel_height = 640, 480
        
        # 调整封面图片大小以适应控件，保持宽高比
        img_width = self.image_cover.GetWidth()
        img_height = self.image_cover.GetHeight()
        aspect_ratio = img_width / img_height
        
        # 根据宽高比调整大小
        if aspect_ratio > panel_width / panel_height:
            # 图片更宽，以宽度为基准
            new_width = panel_width
            new_height = int(new_width / aspect_ratio)
        else:
            # 图片更高，以高度为基准
            new_height = panel_height
            new_width = int(new_height * aspect_ratio)
        
        # 调整图片大小
        self.image_cover.Rescale(new_width, new_height)
        
        # 创建一个带边框的面板来包含图像
        self.image_panel = wx.Panel(self.m_animCtrl1, wx.ID_ANY, wx.DefaultPosition, wx.Size(panel_width, panel_height), wx.BORDER_THEME)
        self.image_panel.SetBackgroundColour(wx.Colour(0, 0, 0))
        image_sizer = wx.BoxSizer(wx.VERTICAL)
        self.bmp = wx.StaticBitmap(self.image_panel, -1, wx.Bitmap(self.image_cover))
        image_sizer.Add(self.bmp, 1, wx.EXPAND | wx.CENTER, 2)
        self.image_panel.SetSizer(image_sizer)
        
        # 在视频控件中添加图像面板
        anim_sizer = wx.BoxSizer(wx.VERTICAL)
        anim_sizer.Add(self.image_panel, 1, wx.EXPAND | wx.ALL, 0)
        self.m_animCtrl1.SetSizer(anim_sizer)

        # 设置窗口标题的图标
        self.icon = wx.Icon(config.ICON_PATH, wx.BITMAP_TYPE_ICO)
        self.SetIcon(self.icon)
    
    def init_camera_params(self):
        """初始化摄像头参数"""
        self.VIDEO_STREAM = config.VIDEO_STREAM
        self.CAMERA_STYLE = config.CAMERA_STYLE_DEFAULT  # False未打开摄像头，True摄像头已打开
        self.cap = None  # 摄像头对象
    
    def bind_events(self):
        """绑定事件处理函数"""
        # 绑定按钮事件
        self.m_choice1.Bind(wx.EVT_BUTTON, self.prepare)
        self.camera_button1.Bind(wx.EVT_BUTTON, self.camera_on)  # 开始检测
        self.off_button3.Bind(wx.EVT_BUTTON, self.off)  # 暂停
        self.off_button4.Bind(wx.EVT_BUTTON, self.exit)  # 退出
        
        # 添加按钮悬停效果
        self.m_choice1.Bind(wx.EVT_ENTER_WINDOW, lambda evt: self.on_button_hover(evt, self.m_choice1, wx.Colour(41, 128, 185)))
        self.m_choice1.Bind(wx.EVT_LEAVE_WINDOW, lambda evt: self.on_button_leave(evt, self.m_choice1, wx.Colour(52, 152, 219)))
        
        self.camera_button1.Bind(wx.EVT_ENTER_WINDOW, lambda evt: self.on_button_hover(evt, self.camera_button1, wx.Colour(39, 174, 96)))
        self.camera_button1.Bind(wx.EVT_LEAVE_WINDOW, lambda evt: self.on_button_leave(evt, self.camera_button1, wx.Colour(46, 204, 113)))
        
        self.off_button3.Bind(wx.EVT_ENTER_WINDOW, lambda evt: self.on_button_hover(evt, self.off_button3, wx.Colour(211, 84, 0)))
        self.off_button3.Bind(wx.EVT_LEAVE_WINDOW, lambda evt: self.on_button_leave(evt, self.off_button3, wx.Colour(230, 126, 34)))
        
        self.off_button4.Bind(wx.EVT_ENTER_WINDOW, lambda evt: self.on_button_hover(evt, self.off_button4, wx.Colour(192, 57, 43)))
        self.off_button4.Bind(wx.EVT_LEAVE_WINDOW, lambda evt: self.on_button_leave(evt, self.off_button4, wx.Colour(231, 76, 60)))
        
        # 绑定窗口关闭事件
        self.Bind(wx.EVT_CLOSE, self.OnClose)
    
    def append_text(self, text):
        """向状态输出区域添加文本
        
        Args:
            text: 要添加的文本
        """
        try:
            # 使用wx.CallAfter确保在主线程中更新UI
            wx.CallAfter(self.m_textCtrl3.AppendText, text)
        except Exception as e:
            # 如果UI组件已被删除，则忽略错误
            print(f"无法更新文本区域: {str(e)}")
    
    def prepare(self, evt):
        """准备加载摄像头"""
        self.append_text(u"加载车载摄像头成功!!!\n")
        speech_utils.speak(speech_utils.MESSAGES["model_loaded"])
    
    def camera_on(self, event):
        """开始检测，启动多线程"""
        # 使用多线程，子线程运行后台的程序，主线程更新前台的UI
        _thread.start_new_thread(self._learning_face, (event,))
        _thread.start_new_thread(self._frequency_counter, (event,))
        _thread.start_new_thread(self._fatigue_alarm, (event,))
    
    def off(self, event):
        """暂停检测，关闭摄像头"""
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
                self.cap = None
                self.CAMERA_STYLE = False
                
                # 重新加载并调整封面图片大小
                self.image_cover = wx.Image(config.COVER, wx.BITMAP_TYPE_ANY)
                
                # 获取图像面板的客户区大小
                panel_size = self.image_panel.GetClientSize()
                panel_width, panel_height = panel_size.GetWidth(), panel_size.GetHeight()
                
                # 如果无法获取有效的尺寸，使用默认尺寸
                if panel_width <= 0 or panel_height <= 0:
                    panel_width, panel_height = 640, 480
                
                # 调整封面图片大小以适应控件，保持宽高比
                img_width = self.image_cover.GetWidth()
                img_height = self.image_cover.GetHeight()
                aspect_ratio = img_width / img_height
                
                # 根据宽高比调整大小
                if aspect_ratio > panel_width / panel_height:
                    # 图片更宽，以宽度为基准
                    new_width = panel_width
                    new_height = int(new_width / aspect_ratio)
                else:
                    # 图片更高，以高度为基准
                    new_height = panel_height
                    new_width = int(new_height * aspect_ratio)
                
                # 调整图片大小
                self.image_cover.Rescale(new_width, new_height)
                
                # 更新显示
                wx.CallAfter(self.bmp.SetBitmap, wx.Bitmap(self.image_cover))
                self.append_text(u"已暂停检测，摄像头已关闭\n")
                wx.CallAfter(self.SetStatusText, u"摄像头已断开 - 检测已暂停")
        except Exception as e:
            self.append_text(f"关闭摄像头错误: {str(e)}\n")
            # 确保无论如何都显示封面图
            wx.CallAfter(self.bmp.SetBitmap, wx.Bitmap(self.image_cover))
    
    def exit(self, evt):
        """退出检测"""
        dlg = wx.MessageDialog(None, u'确定要退出检测吗？', u'操作提示', wx.YES_NO | wx.ICON_QUESTION)
        if (dlg.ShowModal() == wx.ID_YES):
            # 先关闭摄像头，确保线程能够正常退出
            try:
                if self.cap and self.cap.isOpened():
                    self.cap.release()
                    self.cap = None
                    self.CAMERA_STYLE = False
            except:
                pass
            
            # 延迟一小段时间，确保线程有机会退出
            time.sleep(0.5)
            
            self.Destroy()
            speech_utils.speak(speech_utils.MESSAGES["exit"])
            print("检测结束，成功退出程序!!!")
    
    def OnClose(self, evt):
        """窗口关闭事件处理函数"""
        dlg = wx.MessageDialog(None, u'确定要关闭本窗口？', u'操作提示', wx.YES_NO | wx.ICON_QUESTION)
        if (dlg.ShowModal() == wx.ID_YES):
            # 先关闭摄像头，确保线程能够正常退出
            try:
                if self.cap and self.cap.isOpened():
                    self.cap.release()
                    self.cap = None
                    self.CAMERA_STYLE = False
            except:
                pass
            
            # 延迟一小段时间，确保线程有机会退出
            time.sleep(0.5)
            
            self.Destroy()
            speech_utils.speak(speech_utils.MESSAGES["exit"])
            print("检测结束，成功退出程序!!!")
    
    def _learning_face(self, event):
        """人脸检测主循环"""
        # 打开摄像头
        try:
            # 先尝试释放之前可能存在的摄像头资源
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
                
            # 重新打开摄像头
            self.cap = cv2.VideoCapture(self.VIDEO_STREAM, cv2.CAP_DSHOW)  # Windows系统推荐
            
            # 设置摄像头分辨率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if self.cap.isOpened() == True:  # 检查初始化是否成功
                self.CAMERA_STYLE = True
                self.append_text(u"打开摄像头成功!!!\n")
                speech_utils.speak(speech_utils.MESSAGES["camera_success"])
                # 更新状态栏
                self.SetStatusText(u"摄像头已连接 - 正在检测中")
            else:
                speech_utils.speak(speech_utils.MESSAGES["camera_fail"])
                self.append_text(u"摄像头打开失败!!!\n")
                # 显示封面图
                self.bmp.SetBitmap(wx.Bitmap(self.image_cover))
                # 更新状态栏
                self.SetStatusText(u"摄像头连接失败 - 请检查设备")
                return
        except Exception as e:
            self.append_text(f"摄像头初始化错误: {str(e)}\n")
            # 显示封面图
            self.bmp.SetBitmap(wx.Bitmap(self.image_cover))
            # 更新状态栏
            self.SetStatusText(u"摄像头连接错误 - 请检查设备")
            return
        
        # 循环读取视频流
        while (self.cap is not None and self.cap.isOpened()):
            try:
                # 读取一帧
                flag, frame = self.cap.read()
                
                # 检查帧是否成功读取
                if not flag or frame is None:
                    self.append_text(u"视频帧获取失败，尝试重新连接...\n")
                    # 尝试重新连接摄像头
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.VIDEO_STREAM, cv2.CAP_DSHOW)
                    if not self.cap.isOpened():
                        self.append_text(u"重新连接摄像头失败\n")
                        break
                    continue
                    
                start = time.time()
                
                # 检测人脸
                face_detected, shape, leftEye, rightEye, mouth, ear, mar, pose = self.detector.detect_face(frame)
                
                if face_detected:
                    # 绘制人脸特征
                    frame = self.detector.draw_face_features(frame, shape, leftEye, rightEye, mouth)
                    
                    # 更新疲劳检测状态
                    self.analyzer.update_blink(ear)
                    self.analyzer.update_yawn(mar)
                    self.analyzer.update_nod(pose[1])  # pose[1]是欧拉角
                else:
                    # 没有检测到人脸
                    cv2.putText(frame, "No Face", (350, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3, cv2.LINE_AA)
                    self.analyzer.update_no_face()
                
                # 获取当前状态信息
                status = self.analyzer.get_status_info()
                fatigue_level = status["fatigue_level"]
                
                # 更新UI上的统计数据
                try:
                    wx.CallAfter(self.update_stats_ui, status)
                except Exception as e:
                    print(f"更新UI统计数据失败: {str(e)}")
                
                # 在图像上显示疲劳程度
                if fatigue_level == "mild":
                    cv2.putText(frame, "轻度疲劳", (350, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                elif fatigue_level == "moderate":
                    cv2.putText(frame, "中度疲劳", (350, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                elif fatigue_level == "severe":
                    cv2.putText(frame, "重度疲劳", (350, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # 计算FPS
                T = time.time() - start
                fps = 1 / T
                fps_txt = 'FPS: %.2f' % (fps)
                
                # 在图像上显示各种信息，使用更美观的字体和颜色
                # 使用半透明背景使文字更易读
                info_bg = frame.copy()
                cv2.rectangle(info_bg, (5, 5), (300, 180), (0, 0, 0), -1)
                cv2.addWeighted(info_bg, 0.3, frame, 0.7, 0, frame)
                
                # 添加信息文本 - 使用英文显示，避免中文乱码问题
                cv2.putText(frame, "Blinks: {}".format(status["blinks"]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Nods: {}".format(status["nods"]), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Yawns: {}".format(status["yawns"]), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Blink Freq: {:.2f}".format(status["blink_frequency"]), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Fatigue: {}".format(status["fatigue_score"]), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, fps_txt, (frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 转换颜色空间并显示图像
                try:
                    # 获取图像面板的客户区大小（实际可用于显示的区域）
                    panel_size = self.image_panel.GetClientSize()
                    panel_width, panel_height = panel_size.GetWidth(), panel_size.GetHeight()
                    
                    # 如果无法获取有效的面板尺寸，使用创建时设置的固定尺寸
                    if panel_width <= 0 or panel_height <= 0:
                        panel_width, panel_height = 640, 480
                    
                    # 保持原始视频帧的宽高比
                    frame_height, frame_width = frame.shape[:2]
                    aspect_ratio = frame_width / frame_height
                    panel_aspect_ratio = panel_width / panel_height
                    
                    # 创建一个黑色背景，大小与面板一致
                    display_frame = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
                    
                    # 根据宽高比决定如何调整大小
                    if aspect_ratio > panel_aspect_ratio:
                        # 视频帧更宽，以宽度为基准
                        new_width = panel_width
                        new_height = int(new_width / aspect_ratio)
                        # 计算垂直居中的偏移量
                        y_offset = (panel_height - new_height) // 2
                        x_offset = 0
                    else:
                        # 视频帧更高，以高度为基准
                        new_height = panel_height
                        new_width = int(new_height * aspect_ratio)
                        # 计算水平居中的偏移量
                        x_offset = (panel_width - new_width) // 2
                        y_offset = 0
                    
                    # 调整视频帧大小，保持宽高比
                    resized_frame = cv2.resize(frame, (new_width, new_height))
                    
                    # 将调整大小后的帧放置在黑色背景上的正确位置
                    display_frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame
                    
                    # 转换颜色空间
                    image1 = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    pic = wx.Bitmap.FromBuffer(panel_width, panel_height, image1)
                    
                    # 在主线程中更新UI
                    wx.CallAfter(self.bmp.SetBitmap, pic)
                except Exception as e:
                    self.append_text(f"视频帧处理错误: {str(e)}\n")
                    continue
            except Exception as e:
                try:
                    # 使用print记录错误，避免UI线程问题
                    print(f"视频处理错误: {str(e)}")
                    self.append_text(f"视频处理错误: {str(e)}\n")
                except:
                    # 如果UI已被销毁，只打印错误
                    print(f"视频处理错误(UI已关闭): {str(e)}")
                continue
        
        # 释放摄像头
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        # 更新状态栏
        wx.CallAfter(self.SetStatusText, u"摄像头已断开 - 检测已停止")
    
    def update_stats_ui(self, status):
        """更新UI上的统计数据
        
        Args:
            status: 包含当前状态的字典
        """
        try:
            # 更新数据统计面板
            self.blink_value.SetLabel(str(status["blinks"]))
            self.yawn_value.SetLabel(str(status["yawns"]))
            self.nod_value.SetLabel(str(status["nods"]))
            
            # 更新疲劳状态指示器
            fatigue_level = status["fatigue_level"]
            
            # 重置所有指示灯为灰色
            self.normal_indicator.SetBackgroundColour(wx.Colour(200, 200, 200))
            self.mild_indicator.SetBackgroundColour(wx.Colour(200, 200, 200))
            self.moderate_indicator.SetBackgroundColour(wx.Colour(200, 200, 200))
            self.severe_indicator.SetBackgroundColour(wx.Colour(200, 200, 200))
            
            # 根据疲劳等级点亮对应指示灯
            if fatigue_level == "normal":
                self.normal_indicator.SetBackgroundColour(wx.Colour(46, 204, 113))  # 绿色
                wx.CallAfter(self.SetStatusText, u"检测状态: 正常 - 无疲劳迹象")
            elif fatigue_level == "mild":
                self.mild_indicator.SetBackgroundColour(wx.Colour(230, 126, 34))  # 橙色
                wx.CallAfter(self.SetStatusText, u"检测状态: 轻度疲劳 - 请注意休息")
            elif fatigue_level == "moderate":
                self.moderate_indicator.SetBackgroundColour(wx.Colour(231, 76, 60))  # 红色
                wx.CallAfter(self.SetStatusText, u"检测状态: 中度疲劳 - 建议停车休息")
            elif fatigue_level == "severe":
                self.severe_indicator.SetBackgroundColour(wx.Colour(142, 68, 173))  # 紫色
                wx.CallAfter(self.SetStatusText, u"检测状态: 重度疲劳 - 危险! 请立即停车")
            
            # 刷新UI
            self.normal_indicator.Refresh()
            self.mild_indicator.Refresh()
            self.moderate_indicator.Refresh()
            self.severe_indicator.Refresh()
        except Exception as e:
            # 如果UI组件已被删除，则忽略错误
            print(f"更新统计UI失败: {str(e)}")
    
    def _frequency_counter(self, event):
        """频率计算线程"""
        try:
            for i in range(500):  # 限制循环次数，避免无限循环
                if not self.IsShown():  # 检查窗口是否仍然显示
                    break
                self.analyzer.calculate_frequencies()
                self.analyzer.update_fatigue_score()
                time.sleep(0.5)  # 添加短暂延迟，减少CPU使用率
        except Exception as e:
            print(f"频率计算线程错误: {str(e)}")
    
    def _fatigue_alarm(self, event):
        """疲劳警报线程"""
        try:
            for i in range(500):  # 限制循环次数，避免无限循环
                if not self.IsShown():  # 检查窗口是否仍然显示
                    break
                time.sleep(3)  # 每3秒检查一次
                self.analyzer.check_and_alert()
        except Exception as e:
            print(f"疲劳警报线程错误: {str(e)}")
            
    def on_button_hover(self, event, button, color):
        """按钮悬停效果"""
        button.SetBackgroundColour(color)
        button.Refresh()
        
    def on_button_leave(self, event, button, color):
        """按钮离开效果"""
        button.SetBackgroundColour(color)
        button.Refresh()