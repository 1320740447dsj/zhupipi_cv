import sys
import os
import csv
import numpy as np
import vtk
from vtk.util import numpy_support
from vedo import *

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QMessageBox, 
                           QShortcut, QTableWidgetItem, QSplitter)
from PyQt5.QtGui import QKeySequence, QColor, QFont
from PyQt5.QtCore import Qt, pyqtSlot

from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

def safe_mesh_operation(func):
    """装饰器：安全地执行网格操作，捕获常见错误"""
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            print(f"网格操作错误 in {func.__name__}: {e}")
            if hasattr(self, 'statusBar'):
                self.statusBar().showMessage(f"操作错误: {str(e)[:50]}...")
            return None
    return wrapper

class MeshLabelerApp(QMainWindow):
    """
    Mesh Labeler Application
    
    A PyQt5-based application for labeling 3D mesh models.
    Features:
    - 3D model display
    - Label information management
    - File upload/download
    - Label editing
    - Brush mode for mesh labeling
    """
    
    def __init__(self):
        super(MeshLabelerApp, self).__init__()

        # App version
        self.app_version = "1.0.0"

        # Set up the UI
        self.setup_ui()

        # Initialize variables (this will load colormap.csv automatically)
        self.init_variables()

        # Connect signals and slots
        self.connect_signals()

        # Show the window
        self.showMaximized()
        
    def setup_ui(self):
        """Set up the user interface"""
        # Set window title
        self.setWindowTitle(f"网格标记器 - v{self.app_version}")
        
        # Set up larger font for the application
        app_font = QFont()
        app_font.setPointSize(10)  # 增加基础字体大小
        self.setFont(app_font)
        
        # Set central widget
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main layout
        self.main_layout = QtWidgets.QHBoxLayout(self.central_widget)
        
        # Create left panel (3D view)
        self.left_panel = QtWidgets.QFrame()
        self.left_panel.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.left_panel.setMinimumWidth(600)
        
        # Create right panel (controls)
        self.right_panel = QtWidgets.QFrame()
        self.right_panel.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.right_panel.setMaximumWidth(400)
        
        # 为右侧面板设置更大的字体
        right_panel_font = QFont()
        right_panel_font.setPointSize(11)  # 右侧面板使用更大的字体
        self.right_panel.setFont(right_panel_font)
        
        # Add panels to main layout
        self.main_layout.addWidget(self.left_panel, 7)
        self.main_layout.addWidget(self.right_panel, 3)
        
        # Set up left panel (3D view)
        self.setup_3d_view()
        
        # Set up right panel (controls)
        self.setup_control_panel()
        
        # Set up status bar
        self.statusBar().showMessage("准备就绪")
        
    def setup_3d_view(self):
        """Set up the 3D view panel"""
        # Create layout for left panel
        self.left_layout = QtWidgets.QVBoxLayout(self.left_panel)
        
        # Create VTK widget
        self.vtk_frame = QtWidgets.QFrame()
        self.vtk_layout = QtWidgets.QVBoxLayout(self.vtk_frame)
        self.vtkWidget = QVTKRenderWindowInteractor(self.vtk_frame)
        self.vtk_layout.addWidget(self.vtkWidget)
        
        # Create plotter
        self.vp = Plotter(qt_widget=self.vtkWidget)
        
        # 配置交互器以更好地控制鼠标行为
        interactor_style = self.vp.interactor.GetInteractorStyle()
        if interactor_style:
            # 禁用默认的鼠标滚轮缩放行为，我们将通过自己的事件处理器来控制
            try:
                # 如果是vtkInteractorStyleTrackballCamera
                if hasattr(interactor_style, 'SetMouseWheelMotionFactor'):
                    interactor_style.SetMouseWheelMotionFactor(0.0)  # 禁用默认滚轮缩放
            except:
                pass
        
        # 安装事件过滤器来拦截Qt级别的滚轮事件
        self.vtkWidget.installEventFilter(self)
        
        # Add VTK frame to left layout
        self.left_layout.addWidget(self.vtk_frame)
        
    def eventFilter(self, source, event):
        """Qt事件过滤器，用于拦截滚轮事件"""
        if source == self.vtkWidget and event.type() == QtCore.QEvent.Wheel:
            # 检查是否在刷子模式下且按住了Ctrl键
            modifiers = QtWidgets.QApplication.keyboardModifiers()
            ctrl_pressed = modifiers & QtCore.Qt.ControlModifier
            
            if self.brush_mode and ctrl_pressed:
                # 在刷子模式下且按住Ctrl时，拦截滚轮事件并调整刷子大小
                if hasattr(event, 'angleDelta'):
                    delta = event.angleDelta().y()
                    if delta > 0:
                        # 向上滚动，增大刷子
                        self.keyboard_increase_brush_radius(source="wheel")
                    else:
                        # 向下滚动，减小刷子
                        self.keyboard_decrease_brush_radius(source="wheel")
                # 返回True表示事件已被处理，不再传递给其他处理器
                return True
        
        # 对于其他事件，调用父类的事件过滤器
        return super(MeshLabelerApp, self).eventFilter(source, event)
    
    def setup_control_panel(self):
        """Set up the control panel"""
        # Create layout for right panel
        self.right_layout = QtWidgets.QVBoxLayout(self.right_panel)
        
        # Create tabs
        self.tabs = QtWidgets.QTabWidget()
        self.right_layout.addWidget(self.tabs)
        
        # Create tabs for different functions
        self.tab_files = QtWidgets.QWidget()
        self.tab_labels = QtWidgets.QWidget()
        self.tab_edit = QtWidgets.QWidget()
        
        self.tabs.addTab(self.tab_files, "文件")
        self.tabs.addTab(self.tab_labels, "标签")
        self.tabs.addTab(self.tab_edit, "编辑")
        
        # Set up files tab
        self.setup_files_tab()
        
        # Set up labels tab
        self.setup_labels_tab()
        
        # Set up edit tab
        self.setup_edit_tab()
        
    def setup_files_tab(self):
        """Set up the files tab for uploading and downloading files"""
        # Create layout
        self.files_layout = QtWidgets.QVBoxLayout(self.tab_files)
        
        # Upload section
        self.upload_group = QtWidgets.QGroupBox("加载文件")
        upload_group_font = QFont()
        upload_group_font.setPointSize(12)
        upload_group_font.setBold(True)
        self.upload_group.setFont(upload_group_font)
        self.upload_layout = QtWidgets.QVBoxLayout(self.upload_group)
        
        self.upload_button = QtWidgets.QPushButton("加载文件")
        self.upload_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogStart))
        # 设置按钮样式，包括更大的字体和内边距
        self.upload_button.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                font-weight: bold;
                padding: 8px 16px;
                min-height: 30px;
            }
        """)
        self.upload_layout.addWidget(self.upload_button)
        
        self.files_layout.addWidget(self.upload_group)
        
        # Download section
        self.download_group = QtWidgets.QGroupBox("下载结果")
        self.download_group.setFont(upload_group_font)
        self.download_layout = QtWidgets.QVBoxLayout(self.download_group)
        
        self.format_layout = QtWidgets.QHBoxLayout()
        self.format_label = QtWidgets.QLabel("格式:")
        format_label_font = QFont()
        format_label_font.setPointSize(11)
        self.format_label.setFont(format_label_font)
        
        self.format_combo = QtWidgets.QComboBox()
        self.format_combo.addItems(["VTP", "STL"])
        self.format_combo.setStyleSheet("""
            QComboBox {
                font-size: 14px;
                padding: 4px;
                min-height: 24px;
            }
        """)
        self.format_layout.addWidget(self.format_label)
        self.format_layout.addWidget(self.format_combo)
        
        self.download_button = QtWidgets.QPushButton("保存模型")
        self.download_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton))
        self.download_button.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                font-weight: bold;
                padding: 8px 16px;
                min-height: 30px;
            }
        """)
        
        self.download_layout.addLayout(self.format_layout)
        self.download_layout.addWidget(self.download_button)
        
        self.files_layout.addWidget(self.download_group)
        
        # Add stretch to push widgets to the top
        self.files_layout.addStretch()
        
    def setup_labels_tab(self):
        """Set up the labels tab for managing label information"""
        # Create layout
        self.labels_layout = QtWidgets.QVBoxLayout(self.tab_labels)
        
        # Label information section
        self.label_info_group = QtWidgets.QGroupBox("标签信息")
        group_font = QFont()
        group_font.setPointSize(12)
        group_font.setBold(True)
        self.label_info_group.setFont(group_font)
        self.label_info_layout = QtWidgets.QVBoxLayout(self.label_info_group)
        
        # Table for label information
        self.label_table = QtWidgets.QTableWidget()
        self.label_table.setColumnCount(3)
        self.label_table.setHorizontalHeaderLabels(["ID", "颜色", "描述"])
        self.label_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.label_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.label_table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        
        # 隐藏行号
        self.label_table.verticalHeader().setVisible(False)
        
        # 设置表格字体
        table_font = QFont()
        table_font.setPointSize(10)
        self.label_table.setFont(table_font)
        
        # Load colormap button
        self.load_colormap_button = QtWidgets.QPushButton("加载颜色映射CSV文件")
        self.load_colormap_button.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                font-weight: bold;
                padding: 6px 12px;
                min-height: 28px;
            }
        """)
        
        self.label_info_layout.addWidget(self.label_table)
        self.label_info_layout.addWidget(self.load_colormap_button)
        
        self.labels_layout.addWidget(self.label_info_group)
        
        # Active label section
        self.active_label_group = QtWidgets.QGroupBox("活动标签")
        self.active_label_group.setFont(group_font)
        self.active_label_layout = QtWidgets.QHBoxLayout(self.active_label_group)
        
        self.active_label_label = QtWidgets.QLabel("当前标签:")
        label_font = QFont()
        label_font.setPointSize(11)
        self.active_label_label.setFont(label_font)
        
        self.active_label_spin = QtWidgets.QSpinBox()
        self.active_label_spin.setMinimum(0)
        self.active_label_spin.setMaximum(999)
        self.active_label_spin.setStyleSheet("""
            QSpinBox {
                font-size: 16px;
                padding: 4px;
                min-height: 24px;
            }
        """)
        
        self.active_label_color = QtWidgets.QLabel()
        self.active_label_color.setMinimumSize(24, 24)
        self.active_label_color.setMaximumSize(24, 24)
        self.active_label_color.setStyleSheet("background-color: #CCCCCC; border: 1px solid black;")
        
        self.active_label_layout.addWidget(self.active_label_label)
        self.active_label_layout.addWidget(self.active_label_spin)
        self.active_label_layout.addWidget(self.active_label_color)
        
        self.labels_layout.addWidget(self.active_label_group)
        
        # Add stretch to push widgets to the top
        self.labels_layout.addStretch()
        
    def setup_edit_tab(self):
        """Set up the edit tab for editing labels"""
        # Create layout
        self.edit_layout = QtWidgets.QVBoxLayout(self.tab_edit)
        
        # Brush mode section
        self.brush_group = QtWidgets.QGroupBox("刷子模式")
        group_font = QFont()
        group_font.setPointSize(12)
        group_font.setBold(True)
        self.brush_group.setFont(group_font)
        self.brush_layout = QtWidgets.QVBoxLayout(self.brush_group)
        #
        # self.brush_info = QtWidgets.QLabel("按 'B' 进入/退出刷子模式\n移动鼠标时显示红色预览区域\n使用 '+/-' 键或 'Ctrl+滚轮' 调整刷子大小")
        # self.brush_info.setStyleSheet("font-weight: bold; color: #2E8B57; font-size: 11px;")
        #
        self.brush_radius_layout = QtWidgets.QHBoxLayout()
        self.brush_radius_label = QtWidgets.QLabel("刷子半径:")
        label_font = QFont()
        label_font.setPointSize(11)
        self.brush_radius_label.setFont(label_font)
        
        self.brush_radius_spin = QtWidgets.QDoubleSpinBox()
        self.brush_radius_spin.setMinimum(0.1)
        self.brush_radius_spin.setMaximum(10.0)
        self.brush_radius_spin.setValue(1.0)
        self.brush_radius_spin.setSingleStep(0.1)
        self.brush_radius_spin.setStyleSheet("""
            QDoubleSpinBox {
                font-size: 16px;
                padding: 4px;
                min-height: 24px;
            }
        """)
        
        self.brush_radius_layout.addWidget(self.brush_radius_label)
        self.brush_radius_layout.addWidget(self.brush_radius_spin)
        
      #  self.brush_layout.addWidget(self.brush_info)
        self.brush_layout.addLayout(self.brush_radius_layout)
        
        self.edit_layout.addWidget(self.brush_group)
        
        # Label swap section
        self.swap_group = QtWidgets.QGroupBox("标签交换")
        self.swap_group.setFont(group_font)
        self.swap_layout = QtWidgets.QVBoxLayout(self.swap_group)
        
        self.swap_from_layout = QtWidgets.QHBoxLayout()
        self.swap_from_label = QtWidgets.QLabel("源标签:")
        self.swap_from_label.setFont(label_font)
        self.swap_from_spin = QtWidgets.QSpinBox()
        self.swap_from_spin.setMinimum(0)
        self.swap_from_spin.setMaximum(999)
        self.swap_from_spin.setStyleSheet("""
            QSpinBox {
                font-size: 16px;
                padding: 4px;
                min-height: 24px;
            }
        """)
        
        self.swap_from_layout.addWidget(self.swap_from_label)
        self.swap_from_layout.addWidget(self.swap_from_spin)
        
        self.swap_to_layout = QtWidgets.QHBoxLayout()
        self.swap_to_label = QtWidgets.QLabel("目标标签:")
        self.swap_to_label.setFont(label_font)
        self.swap_to_spin = QtWidgets.QSpinBox()
        self.swap_to_spin.setMinimum(0)
        self.swap_to_spin.setMaximum(999)
        self.swap_to_spin.setStyleSheet("""
            QSpinBox {
                font-size: 16px;
                padding: 4px;
                min-height: 24px;
            }
        """)
        
        self.swap_to_layout.addWidget(self.swap_to_label)
        self.swap_to_layout.addWidget(self.swap_to_spin)
        
        self.swap_button = QtWidgets.QPushButton("执行交换")
        self.swap_button.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                font-weight: bold;
                padding: 6px 12px;
                min-height: 28px;
            }
        """)
        
        self.swap_layout.addLayout(self.swap_from_layout)
        self.swap_layout.addLayout(self.swap_to_layout)
        self.swap_layout.addWidget(self.swap_button)
        
        self.edit_layout.addWidget(self.swap_group)
        
        # Keyboard shortcuts section
        self.shortcuts_group = QtWidgets.QGroupBox("键盘快捷键")
        self.shortcuts_group.setFont(group_font)
        self.shortcuts_layout = QtWidgets.QVBoxLayout(self.shortcuts_group)
        
        self.shortcuts_text = QtWidgets.QLabel(
            "B: 开启/关闭刷子模式\n"
            "鼠标移动: 显示红色预览选择区域\n"
            "右键拖拽: 选择区域\n"
            "Shift+左键: 填充功能（连通的背景区域）\n"
            "E: 执行标记（将选中区域标记为当前标签）\n"
            "C: 清除选择\n"
            "Ctrl+右键: 擦除模式\n"
            "Ctrl+Z: 撤销上一步操作\n"
            "Ctrl+鼠标滚轮: 调整刷子大小\n"
            "+ / -: 键盘调整刷子大小\n"
            "S: 显示/隐藏标签ID\n"
            "L: 开启/关闭线框模式"
        )
        self.shortcuts_text.setStyleSheet("font-size: 16px; line-height: 1.4;")
        
        self.shortcuts_layout.addWidget(self.shortcuts_text)
        
        self.edit_layout.addWidget(self.shortcuts_group)
        
        # Add stretch to push widgets to the top
        self.edit_layout.addStretch()
    
    def is_array_empty(self, arr):
        """安全地检查数组是否为空，兼容各种数据类型"""
        if arr is None:
            return True
        
        # 如果是numpy数组
        if hasattr(arr, 'dtype') and hasattr(arr, '__len__'):
            return len(arr) == 0
        # 如果是普通列表
        elif isinstance(arr, list):
            return len(arr) == 0
        # 如果是其他可迭代对象
        elif hasattr(arr, '__iter__'):
            return len(list(arr)) == 0
        # 如果是单个值
        else:
            return False
    
    def init_variables(self):
        """Initialize variables for the application"""
        # Mesh variables
        self.mesh_exist = False
        self.mesh_wireframe_show = False
        self.opened_mesh_path = os.getcwd()
        self.existed_opened_mesh_path = os.getcwd()

        # Initialize with minimal default values (will be overridden by colormap loading)
        self.label_id = np.array([0])
        self.label_colormap = np.array([[255, 162, 143]])
        self.label_description = np.array(["背景"])
        self.vedo_colorlist = [(0, [255, 162, 143])]

        # Brush variables - 移到前面以确保在加载颜色映射前初始化
        self.brush_mode = False
        self.brush_clicked = False
        self.brush_radius = 1.0
        self.brush_selected_pts = []
        self.brush_erased_pts = []
        self.flattened_selected_pt_ids = []
        self.flattened_erased_pt_ids = []
        self.selected_cell_ids = []

        # Active label - 移到前面以确保在加载颜色映射前初始化
        self.brush_active_label = [0]
        self.swap_original_label = [0]
        self.swap_new_label = [0]

        # Control variables - 移到前面以确保在加载颜色映射前初始化
        self.ctrl_pressed = False
        self.shift_pressed = False

        # Try to load colormap.csv automatically at startup
        # If not found, create a basic default colormap
        if not self.load_default_colormap():
            self.create_default_colormap()

        # Create lookup table for colors
        self.lut = build_lut(
            colorlist=self.vedo_colorlist,
            interpolate=False,
        )

    def create_default_colormap(self):
        """Create a default colormap with common colors"""
        # Create a basic colormap with commonly used colors
        default_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        default_colors = [
            [255, 162, 143],  # 背景 - 浅橙色
            [255, 0, 0],      # 红色
            [0, 255, 0],      # 绿色  
            [0, 0, 255],      # 蓝色
            [255, 255, 0],    # 黄色
            [255, 0, 255],    # 品红
            [0, 255, 255],    # 青色
            [128, 0, 128],    # 紫色
            [255, 165, 0],    # 橙色
            [128, 128, 128]   # 灰色
        ]
        default_descriptions = [
            "背景", "标签1", "标签2", "标签3", "标签4", 
            "标签5", "标签6", "标签7", "标签8", "标签9"
        ]

        self.label_id = np.array(default_labels)
        self.label_colormap = np.array(default_colors)
        self.label_description = np.array(default_descriptions)

        # Update the colorlist for vedo
        self.vedo_colorlist = []
        for i_row in range(len(self.label_id)):
            self.vedo_colorlist.append(
                (self.label_id[i_row], self.label_colormap[i_row])
            )

        # Add selection color (gray)
        self.vedo_colorlist.append(
            (np.max(self.label_id) + 1, [169, 169, 169])
        )

        print("使用默认颜色映射（未找到colormap.csv文件）")

    def load_default_colormap(self):
        """Load default colormap.csv file if it exists"""
        default_colormap_path = os.path.join(os.getcwd(), "colormap.csv")
        if os.path.exists(default_colormap_path):
            try:
                self._load_colormap_from_file(default_colormap_path)
                print(f"已加载默认颜色映射文件: {default_colormap_path}")

                # Update the table widget if UI is already set up
                if hasattr(self, 'label_table'):
                    self.update_label_table()

                # Update spinbox ranges if UI is already set up
                if hasattr(self, 'active_label_spin'):
                    self.active_label_spin.setMinimum(min(self.label_id))
                    self.active_label_spin.setMaximum(max(self.label_id))
                    self.swap_from_spin.setMinimum(min(self.label_id))
                    self.swap_from_spin.setMaximum(max(self.label_id))
                    self.swap_to_spin.setMinimum(min(self.label_id))
                    self.swap_to_spin.setMaximum(max(self.label_id))

                return True

            except Exception as e:
                print(f"警告: 无法加载默认颜色映射文件: {str(e)}")
                return False
        else:
            print("未找到colormap.csv文件，将使用内置默认颜色映射")
            return False

    def connect_signals(self):
        """Connect signals and slots"""
        # File tab connections
        self.upload_button.clicked.connect(self.load_mesh)
        self.download_button.clicked.connect(self.save_mesh)
        
        # Label tab connections
        self.load_colormap_button.clicked.connect(self.load_colormap)
        self.active_label_spin.valueChanged.connect(self.active_label_changed)
        
        # Edit tab connections
        self.brush_radius_spin.valueChanged.connect(self.brush_radius_changed)
        self.swap_button.clicked.connect(self.swap_labels)
        self.swap_from_spin.valueChanged.connect(self.swap_from_changed)
        self.swap_to_spin.valueChanged.connect(self.swap_to_changed)
        
        # Tab widget connections
        self.tabs.currentChanged.connect(self.tab_changed)
        
        # Keyboard shortcuts
        self.shortcut_toggle_wireframe = QShortcut(QKeySequence("L"), self)
        self.shortcut_toggle_wireframe.activated.connect(self.toggle_wireframe)
        
        # 刷子大小调整快捷键
        self.shortcut_brush_increase = QShortcut(QKeySequence("+"), self)
        self.shortcut_brush_increase.activated.connect(self.keyboard_increase_brush_radius)
        
        self.shortcut_brush_decrease = QShortcut(QKeySequence("-"), self)
        self.shortcut_brush_decrease.activated.connect(self.keyboard_decrease_brush_radius)
        
        # 刷子大小调整快捷键（数字键盘）
        self.shortcut_brush_increase_numpad = QShortcut(QKeySequence("Ctrl++"), self)
        self.shortcut_brush_increase_numpad.activated.connect(self.keyboard_increase_brush_radius)
        
        self.shortcut_brush_decrease_numpad = QShortcut(QKeySequence("Ctrl+-"), self)
        self.shortcut_brush_decrease_numpad.activated.connect(self.keyboard_decrease_brush_radius)
    
    def add_callbacks(self):
        """Add callbacks for 3D interaction"""
        # Segmentation callbacks
        self.vp.add_callback("KeyPressEvent", self.segmentation_keypress)
        self.vp.add_callback("RightButtonPressEvent", self.brush_onRightClick)
        self.vp.add_callback("MouseMove", self.brush_dragging)
        self.vp.interactor.AddObserver("RightButtonReleaseEvent", self.brush_onRightRelease)
        
        # 彻底移除所有可能的默认滚轮事件处理器
        interactor = self.vp.interactor
        interactor_style = interactor.GetInteractorStyle()
        
        # 移除VTK interactor的默认滚轮事件处理器
        interactor.RemoveObservers("MouseWheelForwardEvent")
        interactor.RemoveObservers("MouseWheelBackwardEvent")
        
        # 如果存在交互器样式，也要禁用其滚轮处理
        if interactor_style:
            try:
                # 禁用交互器样式的滚轮处理
                interactor_style.SetMouseWheelMotionFactor(0.0)
                # 移除交互器样式的滚轮事件处理器
                interactor_style.RemoveObservers("MouseWheelForwardEvent") 
                interactor_style.RemoveObservers("MouseWheelBackwardEvent")
            except:
                pass
        
        # 添加我们的自定义滚轮事件处理器（使用最高优先级）
        interactor.AddObserver("MouseWheelForwardEvent", self.handle_mouse_wheel_forward, 10.0)
        interactor.AddObserver("MouseWheelBackwardEvent", self.handle_mouse_wheel_backward, 10.0)
        
        self.vp.add_callback("LeftButtonPressEvent", self.brush_filling)
        
        # Keyboard callbacks
        self.vp.interactor.AddObserver("KeyPressEvent", self.press_ctrl)
        self.vp.interactor.AddObserver("KeyReleaseEvent", self.release_ctrl)
        self.vp.interactor.AddObserver("KeyPressEvent", self.press_shift)
        self.vp.interactor.AddObserver("KeyReleaseEvent", self.release_shift)
    
    def load_mesh(self):
        """Load a mesh file"""
        self.opened_mesh_path, _ = QFileDialog.getOpenFileName(
            None, "打开文件", self.opened_mesh_path, "*.vtp; *.stl; *.obj; *.ply"
        )
        try:
            if self.opened_mesh_path and self.opened_mesh_path[-4:].lower() in [
                ".vtp", ".stl", ".obj", ".ply"
            ]:
                self.reset_plotters()
                self.plot_mesh()
                self.existed_opened_mesh_path = self.opened_mesh_path
                self.setWindowTitle(
                    f"网格标记器 - v{self.app_version} - {self.existed_opened_mesh_path}"
                )
                self.vtkWidget.setFocus()
        except Exception as e:
            self.show_message_box(f"加载网格时出错: {str(e)}")
    
    def plot_mesh(self):
        """Plot the loaded mesh using vedo"""
        self.mesh_exist = True
        self.mesh = load(self.opened_mesh_path)
        self.mesh_cms = Points(self.mesh.cell_centers())
        self.mesh.linecolor('black').linewidth(0.0)
        self.normals = self.mesh.compute_normals()
        
        # Check if the input mesh has cell array 'Label'; if not, assign zero for all cells
        if not "Label" in self.mesh.celldata.keys():
            self.mesh.celldata["Label"] = np.zeros(
                [self.mesh.ncells], dtype=np.int32
            )
        
        # Update lookup table for colors (in case colormap was loaded)
        self.lut = build_lut(
            colorlist=self.vedo_colorlist,
            interpolate=False,
        )

        self.set_mesh_color()
        
        # Save labels for undo operations
        self.temp_labels = self.mesh.clone().celldata["Label"]
        self.undo_labels = self.mesh.clone().celldata["Label"]
        self.undo_temp_labels = self.mesh.clone().celldata["Label"]
        
        # Make sure 'Label' is the active cell array
        self.mesh.celldata.select("Label")
        
        # Show the mesh
        self.vp.show(self.mesh, interactive=False)
        
        # Add callbacks for interaction
        self.add_callbacks()
        
        # Update status
        self.statusBar().showMessage(f"已加载网格: {self.opened_mesh_path}")
    
    def set_mesh_color(self):
        """Set the mesh color based on labels"""
        if not self.mesh_exist:
            return
            
        self.mesh.cmap(
            input_cmap=self.lut,
            input_array=self.mesh.celldata["Label"],
            on="cells",
            n_colors=len(self.vedo_colorlist),
        )
        self.mesh.mapper().SetScalarRange(0, np.max(self.label_id) + 2)  # Keep space for selection color
    
    def save_mesh(self):
        """Save the labeled mesh"""
        if not self.mesh_exist:
            self.show_message_box("没有可用的网格! 请先加载一个网格!")
            return
            
        try:
            # Clean any active selection
            self.clean_segmentation_selection()
            
            # Handle VTP format
            if self.format_combo.currentText() == "VTP":
                self.save_data_path, _ = QFileDialog.getSaveFileName(
                    None, "保存文件", self.existed_opened_mesh_path[:-4], "*.vtp"
                )
                if self.save_data_path:  # not empty
                    # Create a copy of the mesh for saving
                    self.saved_mesh = self.mesh.clone()
                    file_io.write(self.saved_mesh, self.save_data_path, binary=True)
                    self.existed_opened_mesh_path = self.save_data_path
                    self.setWindowTitle(
                        f"网格标记器 - v{self.app_version} - {self.save_data_path}"
                    )
                    
            # Handle STL format
            elif self.format_combo.currentText() == "STL":
                self.save_data_path, _ = QFileDialog.getSaveFileName(
                    None, "保存文件", self.existed_opened_mesh_path[:-4]
                )
                if self.save_data_path:  # not empty
                    # Create a copy of the mesh for saving
                    self.saved_mesh = self.mesh.clone()
                    
                    # Iterate all unique labels and save each as a separate file
                    label_classes = np.unique(
                        self.saved_mesh.celldata["Label"]
                    ).astype(np.int32)
                    
                    for i_label in label_classes:
                        lb = i_label - 0.5
                        ub = i_label + 0.5
                        tmp_mesh = self.saved_mesh.clone()
                        tmp_mesh = tmp_mesh.threshold(
                            "Label", above=lb, below=ub, on="cells"
                        )
                        file_io.write(
                            tmp_mesh,
                            f"{self.save_data_path}_label_{i_label}.stl",
                            binary=True,
                        )
                    
                    self.existed_opened_mesh_path = self.save_data_path
                    self.setWindowTitle(
                        f"网格标记器 - v{self.app_version} - {self.save_data_path}_label_*.stl"
                    )
            
            # Update status
            self.statusBar().showMessage("文件已保存")
            self.vtkWidget.setFocus()
            
        except Exception as e:
            self.show_message_box(f"保存网格时出错: {str(e)}")
    
    def reset_plotters(self):
        """Reset the 3D view"""
        self.vp.clear()
        self.brush_mode = False
        self.selected_cell_ids = []
    
    def clean_segmentation_selection(self):
        """Clean any active selection in segmentation mode"""
        if not self.mesh_exist:
            return
            
        try:
            self.vp.remove("tmp_select")
            self.vp.render(resetcam=False)
            
            self.brush_selected_pts = []
            self.brush_erased_pts = []
            self.flattened_selected_pt_ids = []
            self.flattened_erased_pt_ids = []
            self.selected_cell_ids = []
            self.brush_mode = False
            
            self.mesh.celldata["Label"] = self.temp_labels  # restore temp_labels
            self.set_mesh_color()
            self.vp.show(self.mesh, resetcam=False)
        except Exception:
            pass  # Ignore errors when cleaning
    
    def selected_pt_ids_to_cell_ids(self, selected_ids):
        """Convert point IDs to cell IDs"""
        if not self.mesh_exist:
            return []
            
        # 处理不同类型的输入（列表、numpy数组、单个值、None）
        if selected_ids is None:
            return []
        
        # 转换输入为列表格式，统一处理
        try:
            # 如果是numpy数组
            if hasattr(selected_ids, 'dtype') and hasattr(selected_ids, '__len__'):
                if len(selected_ids) == 0:
                    return []
                selected_ids = selected_ids.tolist() if hasattr(selected_ids, 'tolist') else list(selected_ids)
            # 如果是普通列表
            elif isinstance(selected_ids, list):
                if len(selected_ids) == 0:
                    return []
            # 如果是单个数值
            elif isinstance(selected_ids, (int, np.integer, float, np.floating)):
                selected_ids = [int(selected_ids)]
            # 如果是numpy标量
            elif np.isscalar(selected_ids):
                selected_ids = [int(selected_ids)]
            else:
                print(f"未知的selected_ids类型: {type(selected_ids)}, 值: {selected_ids}")
                return []
                
        except Exception as e:
            print(f"处理selected_ids时出错: {e}, 类型: {type(selected_ids)}")
            return []
            
        selected_cell_ids = []
        try:
            for i in selected_ids:
                cell_ids = self.mesh.connected_cells(int(i), return_ids=True)
                selected_cell_ids.append(cell_ids)
            
            flat_list = []
            for sublist in selected_cell_ids:
                if hasattr(sublist, '__iter__'):
                    for item in sublist:
                        flat_list.append(item)
                else:
                    flat_list.append(sublist)
            
            return np.unique(np.asarray(flat_list)) if flat_list else []
            
        except Exception as e:
            print(f"转换点ID到单元ID时出错: {e}")
            return []
    
    def toggle_wireframe(self):
        """Toggle wireframe display"""
        if not self.mesh_exist:
            return
            
        self.mesh_wireframe_show = not self.mesh_wireframe_show
        if self.mesh_wireframe_show:
            self.mesh.lw(0.1)
        else:
            self.mesh.lw(0)
        self.vp.render(resetcam=False)
    
    def show_message_box(self, message, title="错误", icon=QMessageBox.Critical):
        """Show a message box with the given message"""
        msgBox = QMessageBox()
        msgBox.setIcon(icon)
        msgBox.setText(message)
        msgBox.setWindowTitle(title)
        msgBox.setStandardButtons(QMessageBox.Ok)
        msgBox.exec()
    
    def load_colormap(self):
        """Load colormap from CSV file"""
        colormap_path, _ = QFileDialog.getOpenFileName(
            None, "选择颜色映射CSV文件", os.getcwd(), "CSV文件 (*.csv);;所有文件 (*.*)"
        )

        if not colormap_path:
            return

        try:
            self._load_colormap_from_file(colormap_path)
            # Update status
            self.statusBar().showMessage(f"已成功加载颜色映射文件: {os.path.basename(colormap_path)}")
            
            # Show success message
            self.show_message_box(
                f"颜色映射文件加载成功！\n\n文件: {os.path.basename(colormap_path)}\n标签数量: {len(self.label_id)}", 
                "加载成功", 
                QMessageBox.Information
            )

        except Exception as e:
            self.show_message_box(f"加载颜色映射文件时出错:\n\n{str(e)}", "加载错误")

    def _load_colormap_from_file(self, colormap_path):
        """Internal method to load colormap from a specific file"""
        # Read the CSV file
        try:
            with open(colormap_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                rows = list(reader)
        except UnicodeDecodeError:
            # Try with different encoding if utf-8 fails
            with open(colormap_path, 'r', newline='', encoding='gbk') as csvfile:
                reader = csv.reader(csvfile)
                rows = list(reader)

        if not rows:
            raise ValueError("CSV文件为空")

        # Check if the CSV has the expected format with headers
        if rows[0][0].lower() == 'label' and len(rows[0]) > 1 and rows[0][1].upper() == 'R':
            # New format with headers: label,R,G,B,Description
            data_rows = rows[1:]  # Skip header row

            if not data_rows:
                raise ValueError("CSV文件只有标题行，没有数据")

            self.label_id = np.array([int(row[0]) for row in data_rows])
            self.label_colormap = np.array([[int(row[1]), int(row[2]), int(row[3])] for row in data_rows])
            if len(rows[0]) > 4:
                self.label_description = np.array([row[4] if len(row) > 4 and row[4] else f"标签 {row[0]}" for row in data_rows])
            else:
                self.label_description = np.array([f"标签 {row[0]}" for row in data_rows])
        else:
            # Old format without headers: label,R,G,B,Description
            # Validate data format
            for i, row in enumerate(rows):
                if len(row) < 4:
                    raise ValueError(f"第{i+1}行数据不完整，至少需要4列（标签ID, R, G, B）")
                try:
                    int(row[0])  # label ID
                    int(row[1])  # R
                    int(row[2])  # G
                    int(row[3])  # B
                except ValueError:
                    raise ValueError(f"第{i+1}行数据格式错误，前4列必须是整数")

            self.label_id = np.array([int(row[0]) for row in rows])
            self.label_colormap = np.array([[int(row[1]), int(row[2]), int(row[3])] for row in rows])
            if len(rows[0]) > 4:
                self.label_description = np.array([row[4] if len(row) > 4 and row[4] else f"标签 {row[0]}" for row in rows])
            else:
                self.label_description = np.array([f"标签 {row[0]}" for row in rows])

        # Validate color values
        if np.any(self.label_colormap < 0) or np.any(self.label_colormap > 255):
            raise ValueError("颜色值必须在0-255范围内")

        # Update the colorlist for vedo
        self.vedo_colorlist = []
        for i_row in range(len(self.label_id)):
            self.vedo_colorlist.append(
                (self.label_id[i_row], self.label_colormap[i_row])
            )

        # Add selection color (gray)
        self.vedo_colorlist.append(
            (np.max(self.label_id) + 1, [169, 169, 169])
        )

        # Update the LUT
        self.lut = build_lut(
            colorlist=self.vedo_colorlist,
            interpolate=False,
        )

        # Update the table widget (only if UI is set up)
        if hasattr(self, 'label_table'):
            self.update_label_table()

        # Update the mesh color if a mesh is loaded
        if self.mesh_exist:
            self.set_mesh_color()
            self.vp.show(self.mesh, resetcam=False)

        # Update the spinbox ranges
        if hasattr(self, 'active_label_spin'):
            self.active_label_spin.setMinimum(min(self.label_id))
            self.active_label_spin.setMaximum(max(self.label_id))
            self.swap_from_spin.setMinimum(min(self.label_id))
            self.swap_from_spin.setMaximum(max(self.label_id))
            self.swap_to_spin.setMinimum(min(self.label_id))
            self.swap_to_spin.setMaximum(max(self.label_id))
    
    def update_label_table(self):
        """Update the label table with current label information"""
        # 确保表格组件存在
        if not hasattr(self, 'label_table'):
            return
            
        # Clear the table
        self.label_table.setRowCount(0)

        # Set the number of rows
        self.label_table.setRowCount(len(self.label_id))

        # Fill the table with label information
        for i_row in range(len(self.label_id)):
            # Label ID
            id_item = QTableWidgetItem(str(self.label_id[i_row]))
            id_item.setTextAlignment(Qt.AlignCenter)
            self.label_table.setItem(i_row, 0, id_item)

            # Color (empty cell with background color)
            color_item = QTableWidgetItem("")
            color_item.setBackground(QColor(
                self.label_colormap[i_row, 0],
                self.label_colormap[i_row, 1],
                self.label_colormap[i_row, 2]
            ))
            self.label_table.setItem(i_row, 1, color_item)

            # Description
            desc_item = QTableWidgetItem(str(self.label_description[i_row]))
            self.label_table.setItem(i_row, 2, desc_item)

        # Resize rows to content
        self.label_table.resizeRowsToContents()

        # Update active label color display (only if UI is ready)
        if hasattr(self, 'active_label_color'):
            self.update_active_label_color()
    
    def update_active_label_color(self):
        """Update the active label color display"""
        # 检查必要的属性是否存在
        if not hasattr(self, 'brush_active_label') or not hasattr(self, 'active_label_color'):
            return
            
        if len(self.label_id) > 0 and self.brush_active_label[0] in self.label_id:
            # Find the index of the active label in the label list
            label_idx = np.where(self.label_id == self.brush_active_label[0])[0][0]

            # Update the color display
            color = self.label_colormap[label_idx]
            self.active_label_color.setStyleSheet(
                f"background-color: rgb({color[0]}, {color[1]}, {color[2]}); border: 1px solid black;"
            )
        else:
            # Default color if no valid label
            self.active_label_color.setStyleSheet("background-color: #CCCCCC; border: 1px solid black;")

    def active_label_changed(self):
        """Handle active label change"""
        # Check if the selected label exists in the label list
        if self.active_label_spin.value() in self.label_id:
            # Update the active label
            self.brush_active_label = [self.active_label_spin.value()]

            # Update the color display
            self.update_active_label_color()

            # Set focus back to the VTK widget
            self.vtkWidget.setFocus()
        else:
            # If the label doesn't exist, show an error and revert to the previous value
            self.show_message_box('标签ID不存在!')
            self.active_label_spin.setValue(self.brush_active_label[0])
    
    def brush_radius_changed(self):
        """Handle brush radius change"""
        self.brush_radius = self.brush_radius_spin.value()
        
        # 如果在刷子模式下，更新状态栏
        if self.brush_mode:
            self.statusBar().showMessage(f"刷子半径设置为: {self.brush_radius:.1f} - 移动鼠标查看预览效果")
    
    def swap_labels(self):
        """Swap labels in the mesh"""
        if not self.mesh_exist:
            self.show_message_box("没有可用的网格! 请先加载一个网格!")
            return
        
        # Update the mesh labels
        self.mesh.celldata["Label"][
            self.mesh.celldata["Label"] == self.swap_original_label[0]
        ] = self.swap_new_label[0]
        
        # Update the backup labels
        self.temp_labels = self.mesh.clone().celldata["Label"]
        
        # Update the mesh color
        self.set_mesh_color()
        self.vp.show(self.mesh, resetcam=False)
        
        # Update status
        self.statusBar().showMessage(f"已将标签 {self.swap_original_label[0]} 更改为 {self.swap_new_label[0]}")
        
        # Set focus back to the VTK widget
        self.vtkWidget.setFocus()
    
    def swap_from_changed(self):
        """Handle swap from label change"""
        if self.swap_from_spin.value() in self.label_id:
            self.swap_original_label = [self.swap_from_spin.value()]
        else:
            self.show_message_box('标签ID不存在!')
            self.swap_from_spin.setValue(self.swap_original_label[0])
    
    def swap_to_changed(self):
        """Handle swap to label change"""
        if self.swap_to_spin.value() in self.label_id:
            self.swap_new_label = [self.swap_to_spin.value()]
        else:
            self.show_message_box('标签ID不存在!')
            self.swap_to_spin.setValue(self.swap_new_label[0])
    
    def tab_changed(self):
        """Handle tab change"""
        current_tab = self.tabs.currentIndex()
        
        if current_tab == 0:  # Files tab
            self.statusBar().showMessage("文件管理 - 加载和保存网格模型")
        elif current_tab == 1:  # Labels tab
            self.statusBar().showMessage("标签管理 - 查看颜色映射和设置活动标签")
        elif current_tab == 2:  # Edit tab
            self.statusBar().showMessage("编辑工具 - 使用刷子工具标记网格或交换标签")
    
    def segmentation_keypress(self, evt):
        """Handle key press events for segmentation"""
        if evt.keypress in ["b", "B"]:
            # Toggle brush mode
            self.brush_mode = not self.brush_mode
            
            if self.brush_mode:
                # Reset selection variables
                self.brush_selected_pts = []
                self.flattened_selected_pt_ids = []
                self.selected_cell_ids = []
                self.ctrl_pressed = False
                self.brush_erased_pts = []
                self.flattened_erased_pt_ids = []
                
                # Update status with detailed instructions
                self.statusBar().showMessage("刷子模式已激活 - 移动鼠标查看预览，右键拖拽选择区域，按E执行标记")
            else:
                # Clean selection when exiting brush mode
                self.clean_segmentation_selection()
                # Update status
                self.statusBar().showMessage("刷子模式已关闭")
                
        elif evt.keypress in ["e", "E"]:
            # Execute labeling (assign active label to selected cells)
            if len(self.selected_cell_ids) > 0:
                # Backup current state for undo
                self.undo_backup()
                
                # Assign active label to selected cells
                self.assign_active_label_to_selection()
                
                # Reset selection variables
                self.brush_clicked = False
                self.ctrl_pressed = False
                self.brush_selected_pts = []
                self.brush_erased_pts = []
                self.flattened_selected_pt_ids = []
                self.flattened_erased_pt_ids = []
                self.selected_cell_ids = []
                
                # Restore labels and update display
                self.mesh.celldata["Label"] = self.temp_labels
                self.set_mesh_color()
                self.vp.show(self.mesh, resetcam=False)
                
                # Update status
                self.statusBar().showMessage("标记执行完成")
            else:
                self.statusBar().showMessage("没有选中的区域需要标记")
                
        elif evt.keypress in ["c", "C"]:
            # Clean selection
            self.undo_backup()
            self.clean_segmentation_selection()
            self.statusBar().showMessage("选择已清除")
            
        elif evt.keypress in ["s", "S"]:
            # Toggle label display (show label IDs as captions)
            self.toggle_label_display()
    
    def toggle_label_display(self):
        """Toggle display of label IDs on the mesh"""
        if not self.mesh_exist:
            return
            
        # Check if captions are already displayed
        if hasattr(self, 'caption_meshes') and self.caption_meshes:
            # Remove captions
            self.vp.remove(self.caption_meshes)
            self.vp.remove(self.label_captions)
            self.vp.render()
            
            # Reset caption variables
            self.caption_meshes = []
            self.label_captions = []
            
            self.statusBar().showMessage("标签显示已关闭")
        else:
            # Initialize caption lists
            self.caption_meshes = []
            self.label_captions = []
            
            # Get unique labels
            unique_labels = np.unique(self.mesh.celldata["Label"])
            
            # Create captions for each label
            for i_label in unique_labels:
                if i_label != 0:  # Skip background label
                    # Create a segment for this label
                    i_seg = self.mesh.clone().threshold('Label', above=i_label-0.5, below=i_label+0.5, on='cells').alpha(0)
                    self.caption_meshes.append(i_seg)
                    
                    if i_seg.ncells > 0:
                        # Create caption
                        i_cap = i_seg.caption(
                            f"{int(i_label)}",
                            point=i_seg.center_of_mass(),
                            size=(0.3, 0.06),
                            padding=0,
                            font="Arial",
                            alpha=1,
                        )
                        i_cap.name = f"cap_{i_label}"
                        self.label_captions.append(i_cap)
            
            # Add captions to the scene
            self.vp.add(self.label_captions).render()
            self.statusBar().showMessage("标签显示已开启")
    
    def undo_backup(self):
        """Backup current state for undo operation"""
        if not self.mesh_exist:
            return
            
        # Save current state
        self.undo_brush_mode = self.brush_mode
        
        # 安全地复制数组，避免空数组的问题
        if self.is_array_empty(self.brush_selected_pts):
            self.undo_brush_selected_pts = []
        else:
            self.undo_brush_selected_pts = self.brush_selected_pts.copy()
            
        if self.is_array_empty(self.brush_erased_pts):
            self.undo_brush_erased_pts = []
        else:
            self.undo_brush_erased_pts = self.brush_erased_pts.copy()
            
        if self.is_array_empty(self.flattened_selected_pt_ids):
            self.undo_flattened_selected_pt_ids = []
        else:
            self.undo_flattened_selected_pt_ids = self.flattened_selected_pt_ids.copy()
            
        if self.is_array_empty(self.flattened_erased_pt_ids):
            self.undo_flattened_erased_pt_ids = []
        else:
            self.undo_flattened_erased_pt_ids = self.flattened_erased_pt_ids.copy()
            
        if self.is_array_empty(self.selected_cell_ids):
            self.undo_selected_cell_ids = []
        else:
            self.undo_selected_cell_ids = self.selected_cell_ids.copy()
            
        if self.is_array_empty(self.mesh.celldata["Label"]):
            self.undo_labels = np.array([])
        else:
            self.undo_labels = self.mesh.celldata["Label"].copy()
            
        if self.is_array_empty(self.temp_labels):
            self.undo_temp_labels = np.array([])
        else:
            self.undo_temp_labels = self.temp_labels.copy()
    
    def undo_recover(self):
        """Recover state from backup for undo operation"""
        if not self.mesh_exist:
            return
            
        # Restore state from backup
        self.brush_mode = self.undo_brush_mode
        self.brush_selected_pts = self.undo_brush_selected_pts.copy()
        self.brush_erased_pts = self.undo_brush_erased_pts.copy()
        self.flattened_selected_pt_ids = self.undo_flattened_selected_pt_ids.copy()
        self.flattened_erased_pt_ids = self.undo_flattened_erased_pt_ids.copy()
        self.selected_cell_ids = self.undo_selected_cell_ids.copy()
        self.mesh.celldata["Label"] = self.undo_labels.copy()
        self.temp_labels = self.undo_temp_labels.copy()
        
        # Update display
        self.vp.show(self.mesh, resetcam=False)
        self.statusBar().showMessage("撤销完成")
    
    def assign_active_label_to_selection(self):
        """Assign the active label to selected cells"""
        if not self.mesh_exist or self.is_array_empty(self.selected_cell_ids):
            return
            
        # Assign active label to selected cells
        self.mesh.celldata["Label"][self.selected_cell_ids] = self.brush_active_label[0]
        
        # Update backup labels
        self.temp_labels = self.mesh.clone().celldata["Label"]
        
        # Update mesh color
        self.set_mesh_color()
        self.vp.show(self.mesh, resetcam=False)
    
    @safe_mesh_operation
    def show_brush_preview(self, picked_point):
        """显示刷子预览效果的统一方法"""
        if not self.mesh_exist or not self.brush_mode or picked_point is None:
            return
            
        # 获取刷子范围内的点
        tmp_pt = self.mesh.closest_point(
            picked_point, radius=self.brush_radius, return_point_id=True
        )
        
        # 确保 tmp_pt 是有效的
        if tmp_pt is None:
            return
            
        # 转换为网格单元并显示预览
        tmp_cells = self.selected_pt_ids_to_cell_ids(tmp_pt)
        if len(tmp_cells) > 0:
            mesh_center_points = self.mesh.cell_centers()
            tmp_cell_pts = mesh_center_points[tmp_cells]
            tmp_pts = Points(tmp_cell_pts).c("red").pickable(False)
            tmp_pts.name = "tmp_select"
            self.vp.remove("tmp_select").add(tmp_pts).render()
    
    def brush_onRightClick(self, evt):
        """Handle right click for brush"""
        if not self.mesh_exist or not self.brush_mode:
            return
            
        if not self.brush_clicked:
            # Start brush selection
            self.brush_clicked = True
            self.vp.show(resetcam=False)
            
            # Get picked point
            p = evt.picked3d
            if p is None:
                return
            
            # Show immediate preview when right click starts (参考原版逻辑)
            self.show_brush_preview(p)
        
            # Backup current state for undo
            self.undo_backup()
            
            # Get points within brush radius
            pt_ids = self.mesh.closest_point(
                p, radius=self.brush_radius, return_point_id=True
            )
            
            # 确保 pt_ids 是有效的
            if pt_ids is None:
                return
            
            # 如果 pt_ids 是单个值，转换为列表
            if not isinstance(pt_ids, (list, np.ndarray)):
                pt_ids = [pt_ids]
            
            # Handle selection based on ctrl key
            if not self.ctrl_pressed:
                # Add to selection
                if self.is_array_empty(self.flattened_selected_pt_ids):
                    self.brush_selected_pts = []
                else:
                    self.brush_selected_pts = list(self.flattened_selected_pt_ids)
                for i in pt_ids:
                    if i not in self.brush_selected_pts:
                        self.brush_selected_pts.append(i)
                self.flattened_selected_pt_ids = np.asarray(
                    self.brush_selected_pts, dtype=np.int32
                )
            else:
                # Add to erased points
                if self.is_array_empty(self.flattened_erased_pt_ids):
                    self.brush_erased_pts = []
                else:
                    self.brush_erased_pts = list(self.flattened_erased_pt_ids)
                for i in pt_ids:
                    if i not in self.brush_erased_pts:
                        self.brush_erased_pts.append(i)
                self.flattened_erased_pt_ids = np.asarray(
                    self.brush_erased_pts, dtype=np.int32
                )
            
            # Final selection = selected pts - erased pts
            if self.is_array_empty(self.flattened_selected_pt_ids):
                final_selection = []
            else:
                final_selection = [
                    i
                    for i in list(self.flattened_selected_pt_ids)
                    if i not in list(self.flattened_erased_pt_ids)
                ]
            
            self.flattened_selected_pt_ids = np.asarray(final_selection, dtype=np.int32)
            
            # Convert point IDs to cell IDs
            self.selected_cell_ids = self.selected_pt_ids_to_cell_ids(
                self.flattened_selected_pt_ids
            )
            
            # Update mesh display
            self.mesh.celldata["Label"] = self.temp_labels
            if len(self.selected_cell_ids) > 0:
                # Assign selection color to selected cells
                self.mesh.celldata["Label"][self.selected_cell_ids] = np.max(self.label_id) + 1
            
            # Update display
            self.set_mesh_color()
            self.vp.show(self.mesh, resetcam=False)
    
    def brush_onRightRelease(self, iren, evt):
        """Handle right button release for brush"""
        if self.mesh_exist and self.brush_mode and self.brush_clicked:
            # End brush selection
            self.brush_clicked = False
    
    def brush_dragging(self, evt):
        """Handle mouse dragging for brush"""
        if not self.mesh_exist or not self.brush_mode:
            return

        # Get picked point
        p = evt.picked3d
        if p is None:
            return

        if self.brush_clicked:
            # When dragging with right button pressed

            # Show temporary selection preview
            self.show_brush_preview(p)
            
            # Get points within brush radius
            pt_ids = self.mesh.closest_point(
                p, radius=self.brush_radius, return_point_id=True
            )
            
            # 确保 pt_ids 是有效的
            if pt_ids is None:
                return
            
            # 如果 pt_ids 是单个值，转换为列表
            if not isinstance(pt_ids, (list, np.ndarray)):
                pt_ids = [pt_ids]
            
            # Handle selection based on ctrl key
            if not self.ctrl_pressed:
                # Add to selection
                if self.is_array_empty(self.flattened_selected_pt_ids):
                    self.brush_selected_pts = []
                else:
                    self.brush_selected_pts = list(self.flattened_selected_pt_ids)
                for i in pt_ids:
                    if i not in self.brush_selected_pts:
                        self.brush_selected_pts.append(i)
                self.flattened_selected_pt_ids = np.asarray(
                    self.brush_selected_pts, dtype=np.int32
                )
            else:
                # Add to erased points
                if self.is_array_empty(self.flattened_erased_pt_ids):
                    self.brush_erased_pts = []
                else:
                    self.brush_erased_pts = list(self.flattened_erased_pt_ids)
                for i in pt_ids:
                    if i not in self.brush_erased_pts:
                        self.brush_erased_pts.append(i)
                self.flattened_erased_pt_ids = np.asarray(
                    self.brush_erased_pts, dtype=np.int32
                )
            
            # Final selection = selected pts - erased pts
            if self.is_array_empty(self.flattened_selected_pt_ids):
                final_selection = []
            else:
                final_selection = [
                    i
                    for i in list(self.flattened_selected_pt_ids)
                    if i not in list(self.flattened_erased_pt_ids)
                ]
            
            self.flattened_selected_pt_ids = np.asarray(final_selection, dtype=np.int32)
            
            # Convert point IDs to cell IDs
            self.selected_cell_ids = self.selected_pt_ids_to_cell_ids(
                self.flattened_selected_pt_ids
            )
            
            # Update mesh display
            self.mesh.celldata["Label"] = self.temp_labels
            if len(self.selected_cell_ids) > 0:
                # Assign selection color to selected cells
                self.mesh.celldata["Label"][self.selected_cell_ids] = np.max(self.label_id) + 1
            
            # Update display
            self.set_mesh_color()
            self.vp.show(self.mesh, resetcam=False)
        else:
            # When just hovering with brush mode on (新增的预览功能)
            
            # Show temporary selection preview even when not clicking
            self.show_brush_preview(p)
    
    def handle_mouse_wheel_forward(self, obj, evt):
        """Custom VTK mouse wheel forward event handler"""
        # 检查是否按住了Ctrl键
        ctrl_pressed = obj.GetControlKey() == 1
        
        if self.brush_mode and ctrl_pressed:
            # 刷子模式且按住Ctrl时，事件已经被Qt事件过滤器处理了，直接返回
            return
        elif self.mesh_exist:
            # 否则执行正常的缩放功能
            camera = self.vp.camera
            camera.Zoom(1.1)  # 放大10%
            self.vp.render()
    
    def handle_mouse_wheel_backward(self, obj, evt):
        """Custom VTK mouse wheel backward event handler"""
        # 检查是否按住了Ctrl键
        ctrl_pressed = obj.GetControlKey() == 1
        
        if self.brush_mode and ctrl_pressed:
            # 刷子模式且按住Ctrl时，事件已经被Qt事件过滤器处理了，直接返回
            return
        elif self.mesh_exist:
            # 否则执行正常的缩放功能
            camera = self.vp.camera
            camera.Zoom(0.9)  # 缩小10%
            self.vp.render()
    
    def press_shift(self, obj, evt):
        """Handle shift key press"""
        if obj.GetShiftKey() == 1:
            self.shift_pressed = True
    
    def release_shift(self, obj, evt):
        """Handle shift key release"""
        if obj.GetShiftKey() == 0:
            self.shift_pressed = False
    
    def brush_filling(self, evt):
        """Handle brush filling (shift + left click)"""
        if not self.mesh_exist or not self.brush_mode or not self.shift_pressed:
            return
            
        # Get picked point
        p = evt.picked3d
        if p is None:
            return
        
        print('Shift+左键 - 开始填充操作', p)
        
        # Find the closest point
        selected_filling_pt_id = self.mesh.closest_point(p, return_point_id=True)
        
        # 确保找到了有效的点
        if selected_filling_pt_id is None:
            self.statusBar().showMessage("未找到有效的填充起始点")
            return
            
        # 如果返回的是数组，取第一个元素
        if isinstance(selected_filling_pt_id, (list, np.ndarray)):
            if len(selected_filling_pt_id) == 0:
                self.statusBar().showMessage("未找到有效的填充起始点")
                return
            selected_filling_pt_id = selected_filling_pt_id[0]
        
        selected_filling_pt_ids = [selected_filling_pt_id]
        
        # Initialize filling variables
        selected_filling_cell_ids = []
        tmp_selected_filling_cell_ids = []
        
        # Get connected cells with label 0 (background)
        i_cell_ids = self.mesh.connected_cells(selected_filling_pt_id, return_ids=True)
        for j in i_cell_ids:
            if self.mesh.celldata["Label"][j] == 0 and j not in selected_filling_cell_ids:
                selected_filling_cell_ids.append(j)
        
        # Get mesh cells
        mesh_cells = self.mesh.cells()
        mesh_cells = np.array(mesh_cells)
        
        # Iterative filling algorithm (flood fill)
        while len(selected_filling_cell_ids) != len(tmp_selected_filling_cell_ids):
            # Find new cells to process
            diff_cells = list(set(selected_filling_cell_ids) - set(tmp_selected_filling_cell_ids))
            
            # Clone the current selection
            tmp_selected_filling_cell_ids = selected_filling_cell_ids.copy()
            
            # Get points of new cells
            next_round_cell_pts = mesh_cells[diff_cells]
            next_round_cell_pts = np.unique(next_round_cell_pts)
            
            # Process each point
            for i_pt in next_round_cell_pts:
                if i_pt not in selected_filling_pt_ids:
                    selected_filling_pt_ids.append(i_pt)
                    ii_cell_ids = self.mesh.connected_cells(i_pt, return_ids=True)
                    for j in ii_cell_ids:
                        if self.mesh.celldata["Label"][j] == 0 and j not in selected_filling_cell_ids:
                            selected_filling_cell_ids.append(j)
        
        # Update selection
        self.selected_cell_ids = selected_filling_cell_ids
        
        # Update mesh display
        self.mesh.celldata["Label"] = self.temp_labels
        if len(self.selected_cell_ids) > 0:
            # Assign selection color to selected cells
            self.mesh.celldata["Label"][self.selected_cell_ids] = np.max(self.label_id) + 1
        
        # Update display
        self.set_mesh_color()
        self.vp.show(self.mesh, resetcam=False)
        
        # Update status
        self.statusBar().showMessage(f"填充完成 - 选中了 {len(self.selected_cell_ids)} 个网格单元")
    
    def press_ctrl(self, obj, evt):
        """Handle ctrl key press"""
        if obj.GetControlKey() == 1:
            self.ctrl_pressed = True
            
            # Reset erased points
            self.brush_erased_pts = []
            self.flattened_erased_pt_ids = []
            
            # Handle Ctrl+Z for undo
            if obj.GetKeySym() in ["z", "Z"]:
                self.undo_recover()
    
    def release_ctrl(self, obj, evt):
        """Handle ctrl key release"""
        if obj.GetControlKey() == 0:
            self.ctrl_pressed = False
            
            # Reset erased points
            self.brush_erased_pts = []
            self.flattened_erased_pt_ids = []

    def keyboard_increase_brush_radius(self, source="keyboard"):
        """通过键盘快捷键或滚轮增加刷子半径"""
        if not self.mesh_exist or not self.brush_mode:
            return
            
        # 增加刷子半径
        old_radius = self.brush_radius
        self.brush_radius += 0.1
        if self.brush_radius > 10.0:  # 限制最大值
            self.brush_radius = 10.0
            
        # 同步更新界面上的数值
        if hasattr(self, 'brush_radius_spin'):
            self.brush_radius_spin.setValue(self.brush_radius)
        
        # 更新状态栏显示
        if old_radius != self.brush_radius:
            if source == "wheel":
                self.statusBar().showMessage(f"刷子半径增加到: {self.brush_radius:.1f} (Ctrl+滚轮)")
            else:
                self.statusBar().showMessage(f"刷子半径增加到: {self.brush_radius:.1f} (+键)")
        else:
            self.statusBar().showMessage(f"刷子半径已达到最大值: {self.brush_radius:.1f}")
    
    def keyboard_decrease_brush_radius(self, source="keyboard"):
        """通过键盘快捷键或滚轮减小刷子半径"""
        if not self.mesh_exist or not self.brush_mode:
            return
            
        # 减小刷子半径
        old_radius = self.brush_radius
        if self.brush_radius > 0.1:
            self.brush_radius -= 0.1
        else:
            self.brush_radius = 0.1
        
        # 同步更新界面上的数值
        if hasattr(self, 'brush_radius_spin'):
            self.brush_radius_spin.setValue(self.brush_radius)
        
        # 更新状态栏显示
        if old_radius != self.brush_radius:
            if source == "wheel":
                self.statusBar().showMessage(f"刷子半径减小到: {self.brush_radius:.1f} (Ctrl+滚轮)")
            else:
                self.statusBar().showMessage(f"刷子半径减小到: {self.brush_radius:.1f} (-键)")
        else:
            self.statusBar().showMessage(f"刷子半径已达到最小值: {self.brush_radius:.1f}")

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern style
    window = MeshLabelerApp()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 