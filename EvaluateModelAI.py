import os
import sys
import cv2
import time
import datetime
from pathlib import Path
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QPushButton, QFileDialog,
    QSlider, QComboBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QProgressBar, QTextEdit, QListWidget, QMessageBox
)

# Configuration Constants
FOLDER_TO_CLASS = {
    "Land": "0-Land",
    "Forward": "1-Forward",
    "Backward": "2-Backward",
    "Right": "3-Right",
    "Left": "4-Left",
    "Takeoff": "5-Takeoff",
    "Down": "Down",
    "Hover": "Hover",
    "Up": "Up"
}

class VideoWorker(QThread):
    # Signals to communicate with the Main UI
    frame_ready = Signal(QImage, dict)
    stats_updated = Signal(str, bool, str) # folder_name, is_detected, predicted_class
    log_message = Signal(str)
    finished_folder = Signal(dict) # Final stats dictionary

    def __init__(self, model_path, source_type, source_val, conf_thresh, iou_thresh):
        super().__init__()
        self.model_path = model_path
        self.source_type = source_type # "webcam", "video", "folder"
        self.source_val = source_val   # index (int), file_path (str), folder_path (str)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.running = True
        self.paused = False

    def run(self):
        self.log_message.emit("[INFO] Starting worker thread...")
        
        # Load YOLO Model inside the thread to avoid UI freeze
        try:
            from ultralytics import YOLO
            model = YOLO(self.model_path)
            self.log_message.emit(f"[OK] Model loaded: {Path(self.model_path).name}")
        except Exception as e:
            self.log_message.emit(f"[ERR] Error loading model: {str(e)}")
            return

        if self.source_type == "webcam":
            self.run_webcam_or_video(model, int(self.source_val))
        elif self.source_type == "video":
            self.run_webcam_or_video(model, str(self.source_val))
        elif self.source_type == "folder":
            self.run_folder_validation(model, str(self.source_val))

    def run_webcam_or_video(self, model, source):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self.log_message.emit(f"[ERR] Cannot open video source: {source}")
            return

        self.log_message.emit("[OK] Starting video stream...")
        
        prev_time = time.time()
        while self.running:
            if self.paused:
                self.msleep(100)
                continue

            ret, frame = cap.read()
            if not ret:
                self.log_message.emit("[INFO] Video stream ended or camera disconnected.")
                break

            # YOLO inference
            results = model(frame, conf=self.conf_thresh, iou=self.iou_thresh, verbose=False)
            
            # FPS Calculation
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
            prev_time = curr_time

            stats_summary = {"fps": fps, "detections": []}
            
            if len(results) > 0:
                result = results[0]
                annotated_frame = result.plot()
                
                # Extract detected classes for live visual counters
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = model.names[class_id]
                        stats_summary["detections"].append((class_name, conf))
            else:
                annotated_frame = frame

            # Convert frame to QImage and emit
            rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            self.frame_ready.emit(q_image, stats_summary)
            
            # Limit thread speed
            self.msleep(15)

        cap.release()
        self.log_message.emit("[INFO] Released video source.")

    def run_folder_validation(self, model, folder_path):
        self.log_message.emit(f"[INFO] Starting dataset validation at: {folder_path}")
        dataset_path = Path(folder_path)
        
        folder_stats = {} # {folder_name: {total, detected, correct}}
        for folder_name in FOLDER_TO_CLASS.keys():
            folder_stats[folder_name] = {"total": 0, "detected": 0, "correct": 0}

        # Gather all images
        image_tasks = []
        for folder_name in FOLDER_TO_CLASS.keys():
            sub_dir = dataset_path / folder_name
            if sub_dir.exists():
                images = [f for f in sub_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
                for img_path in images:
                    image_tasks.append((folder_name, img_path))

        if not image_tasks:
            self.log_message.emit("[ERR] No images found in dataset!")
            return

        self.log_message.emit(f"[INFO] Found {len(image_tasks)} images. Scanning...")
        
        for idx, (folder_name, img_path) in enumerate(image_tasks):
            if not self.running:
                break
                
            while self.paused:
                self.msleep(100)

            # Load and run model
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            results = model(frame, conf=self.conf_thresh, iou=self.iou_thresh, verbose=False)
            
            is_detected = False
            pred_class_name = ""
            annotated_frame = frame

            if len(results) > 0:
                result = results[0]
                annotated_frame = result.plot()
                boxes = result.boxes
                
                if boxes is not None and len(boxes) > 0:
                    is_detected = True
                    # Get box with highest confidence
                    best_idx = boxes.conf.argmax()
                    pred_class_idx = int(boxes.cls[best_idx])
                    pred_class_name = model.names[pred_class_idx]

            # Update local statistics
            expected_class_name = FOLDER_TO_CLASS[folder_name]
            folder_stats[folder_name]["total"] += 1
            if is_detected:
                folder_stats[folder_name]["detected"] += 1
                if pred_class_name == expected_class_name:
                    folder_stats[folder_name]["correct"] += 1

            # Emit signal to update UI table in real time
            self.stats_updated.emit(folder_name, is_detected, pred_class_name)

            # Convert image to QImage and emit
            rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Simulated progress info
            progress_summary = {
                "fps": 0,
                "current_image": img_path.name,
                "current_folder": folder_name,
                "index": idx + 1,
                "total": len(image_tasks)
            }
            self.frame_ready.emit(q_image, progress_summary)
            
            # Short sleep to make it a smooth slideshow animation
            self.msleep(150)

        self.finished_folder.emit(folder_stats)
        self.log_message.emit("[SUCCESS] Dataset validation completed!")

    def stop(self):
        self.running = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv8 Gesture Model Tester")
        self.resize(1200, 750)
        
        self.model_path = str(Path(__file__).resolve().parent / "best.pt")
        self.worker = None
        self.current_frame = None  # Cache for screenshotting
        
        # Setup modern dark style sheet (Catppuccin Mocha inspired)
        self.apply_stylesheet()
        
        # Build UI Structure
        self.init_ui()
        
        # Auto load default model
        self.update_model_ui()

    def apply_stylesheet(self):
        qss = """
        QMainWindow {
            background-color: #1e1e2e;
        }
        QWidget {
            color: #cdd6f4;
            font-family: "Segoe UI", Arial, sans-serif;
            font-size: 13px;
        }
        QGroupBox {
            border: 1px solid #3f3f5a;
            border-radius: 8px;
            margin-top: 15px;
            font-weight: bold;
            color: #89b4fa;
            background-color: #252538;
            padding: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 15px;
            padding: 0 5px;
        }
        QPushButton {
            background-color: #313244;
            border: 1px solid #45475a;
            border-radius: 6px;
            padding: 6px 12px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #45475a;
            border-color: #89b4fa;
        }
        QPushButton:pressed {
            background-color: #585b70;
        }
        QLineEdit {
            background-color: #181825;
            border: 1px solid #45475a;
            border-radius: 5px;
            padding: 4px;
            color: #cdd6f4;
        }
        QComboBox {
            background-color: #181825;
            border: 1px solid #45475a;
            border-radius: 5px;
            padding: 4px;
            color: #cdd6f4;
        }
        QSlider::groove:horizontal {
            border: 1px solid #45475a;
            height: 6px;
            background: #181825;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #89b4fa;
            border: 1px solid #74c7ec;
            width: 14px;
            margin: -4px 0;
            border-radius: 7px;
        }
        QTabWidget::pane {
            border: 1px solid #3f3f5a;
            background-color: #252538;
            border-radius: 6px;
        }
        QTabBar::tab {
            background-color: #181825;
            border: 1px solid #3f3f5a;
            padding: 6px 15px;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            color: #a6adc8;
        }
        QTabBar::tab:selected {
            background-color: #252538;
            border-bottom-color: #252538;
            color: #89b4fa;
            font-weight: bold;
        }
        QTableWidget {
            background-color: #181825;
            gridline-color: #313244;
            border: 1px solid #3f3f5a;
            border-radius: 5px;
        }
        QHeaderView::section {
            background-color: #313244;
            color: #cdd6f4;
            padding: 4px;
            border: 1px solid #3f3f5a;
            font-weight: bold;
        }
        QProgressBar {
            border: 1px solid #45475a;
            border-radius: 4px;
            text-align: center;
            background-color: #181825;
        }
        QProgressBar::chunk {
            background-color: #a6e3a1;
        }
        QTextEdit {
            background-color: #11111b;
            border: 1px solid #313244;
            border-radius: 5px;
            font-family: Consolas, Monaco, monospace;
            font-size: 12px;
            color: #a6e3a1;
        }
        QListWidget {
            background-color: #181825;
            border: 1px solid #3f3f5a;
            border-radius: 5px;
        }
        """
        self.setStyleSheet(qss)

    def init_ui(self):
        # Main Layout: Three Columns (Left - Center - Right)
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(12)

        # ==================== COLUMN 1: LEFT CONFIG ====================
        left_panel = QWidget()
        left_panel.setFixedWidth(290)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        # 1. Model Configuration Group
        model_group = QGroupBox("MODEL CONFIGURATION")
        model_layout = QVBoxLayout(model_group)
        
        self.lbl_model_name = QLabel("Model: None")
        self.lbl_model_name.setWordWrap(True)
        btn_select_model = QPushButton("Select Model File (.pt)")
        btn_select_model.clicked.connect(self.browse_model)
        
        model_layout.addWidget(self.lbl_model_name)
        model_layout.addWidget(btn_select_model)
        left_layout.addWidget(model_group)

        # 2. Source Selector Group
        source_group = QGroupBox("TEST DATA SOURCE")
        source_layout = QVBoxLayout(source_group)
        
        self.source_tabs = QTabWidget()
        
        # Tab A: Webcam
        tab_webcam = QWidget()
        webcam_layout = QVBoxLayout(tab_webcam)
        webcam_layout.addWidget(QLabel("Select Webcam Index:"))
        self.cb_webcam_idx = QComboBox()
        self.cb_webcam_idx.addItems(["0", "1", "2", "3"])
        webcam_layout.addWidget(self.cb_webcam_idx)
        webcam_layout.addStretch()
        self.source_tabs.addTab(tab_webcam, "Webcam")

        # Tab B: Video file
        tab_video = QWidget()
        video_layout = QVBoxLayout(tab_video)
        video_layout.addWidget(QLabel("Select Video File (.mp4/.avi):"))
        self.lbl_video_path = QLabel("No file selected")
        self.lbl_video_path.setWordWrap(True)
        btn_select_video = QPushButton("Select Video File")
        btn_select_video.clicked.connect(self.browse_video)
        video_layout.addWidget(self.lbl_video_path)
        video_layout.addWidget(btn_select_video)
        video_layout.addStretch()
        self.source_tabs.addTab(tab_video, "Video")

        # Tab C: Dataset Folder
        tab_folder = QWidget()
        folder_layout = QVBoxLayout(tab_folder)
        folder_layout.addWidget(QLabel("Select Dataset Folder (dist/dataset):"))
        self.lbl_folder_path = QLabel("dist/dataset")
        self.lbl_folder_path.setWordWrap(True)
        btn_select_folder = QPushButton("Select Folder")
        btn_select_folder.clicked.connect(self.browse_folder)
        folder_layout.addWidget(self.lbl_folder_path)
        folder_layout.addWidget(btn_select_folder)
        folder_layout.addStretch()
        self.source_tabs.addTab(tab_folder, "Dataset")
        
        # Set default tab to Folder Dataset
        self.source_tabs.setCurrentIndex(2)

        source_layout.addWidget(self.source_tabs)
        left_layout.addWidget(source_group)

        # 3. Parameters Group
        param_group = QGroupBox("DETECTION PARAMETERS")
        param_layout = QGridLayout(param_group)
        
        # Confidence
        param_layout.addWidget(QLabel("Conf (Confidence Threshold):"), 0, 0)
        self.lbl_conf_val = QLabel("0.25")
        param_layout.addWidget(self.lbl_conf_val, 0, 1, Qt.AlignRight)
        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(5, 95)
        self.slider_conf.setValue(25)
        self.slider_conf.valueChanged.connect(self.on_slider_changed)
        param_layout.addWidget(self.slider_conf, 1, 0, 1, 2)

        # IOU
        param_layout.addWidget(QLabel("IOU (Overlap Threshold):"), 2, 0)
        self.lbl_iou_val = QLabel("0.45")
        param_layout.addWidget(self.lbl_iou_val, 2, 1, Qt.AlignRight)
        self.slider_iou = QSlider(Qt.Horizontal)
        self.slider_iou.setRange(5, 95)
        self.slider_iou.setValue(45)
        self.slider_iou.valueChanged.connect(self.on_slider_changed)
        param_layout.addWidget(self.slider_iou, 3, 0, 1, 2)
        
        left_layout.addWidget(param_group)

        # 4. Action Buttons
        self.btn_start = QPushButton("START DETECTION")
        self.btn_start.setObjectName("btn_start")
        self.btn_start.setStyleSheet("background-color: #a6e3a1; color: #11111b; font-size: 14px;")
        self.btn_start.clicked.connect(self.start_detection)

        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("background-color: #f38ba8; color: #11111b; font-size: 14px;")
        self.btn_stop.clicked.connect(self.stop_detection)

        left_layout.addWidget(self.btn_start)
        left_layout.addWidget(self.btn_stop)
        left_layout.addStretch()

        main_layout.addWidget(left_panel)

        # ==================== COLUMN 2: CENTER LIVE VIEW ====================
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)

        # Screen View Window
        view_group = QGroupBox("MODEL INFERENCE VIEWER")
        view_layout = QVBoxLayout(view_group)
        
        self.lbl_view = QLabel()
        self.lbl_view.setMinimumSize(540, 400)
        self.lbl_view.setStyleSheet("background-color: #11111b; border: 2px dashed #313244; border-radius: 6px;")
        self.lbl_view.setAlignment(Qt.AlignCenter)
        self.lbl_view.setText("LIVE STREAM - NO DATA\n\nSelect a source on the left and click 'START DETECTION'")
        view_layout.addWidget(self.lbl_view)

        # Controls under viewport
        controls_layout = QHBoxLayout()
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self.toggle_pause)
        
        self.btn_screenshot = QPushButton("Take Screenshot")
        self.btn_screenshot.setEnabled(False)
        self.btn_screenshot.clicked.connect(self.take_screenshot)
        
        self.lbl_hud_fps = QLabel("FPS: --")
        self.lbl_hud_info = QLabel("Frame: --")
        
        controls_layout.addWidget(self.btn_pause)
        controls_layout.addWidget(self.btn_screenshot)
        controls_layout.addStretch()
        controls_layout.addWidget(self.lbl_hud_info)
        controls_layout.addWidget(self.lbl_hud_fps)
        
        view_layout.addLayout(controls_layout)
        center_layout.addWidget(view_group)

        # System Log
        log_group = QGroupBox("CONSOLE LOG")
        log_layout = QVBoxLayout(log_group)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFixedHeight(120)
        log_layout.addWidget(self.txt_log)
        center_layout.addWidget(log_group)

        main_layout.addWidget(center_panel, stretch=2)

        # ==================== COLUMN 3: RIGHT STATS & HISTORY ====================
        right_panel = QWidget()
        right_panel.setFixedWidth(360)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # A. Live statistics table
        stats_group = QGroupBox("MODEL EVALUATION DASHBOARD")
        stats_layout = QVBoxLayout(stats_group)
        
        self.tbl_stats = QTableWidget()
        self.tbl_stats.setColumnCount(4)
        self.tbl_stats.setHorizontalHeaderLabels(["Gesture", "Total", "Detected", "Accuracy %"])
        self.tbl_stats.horizontalHeader().setStretchLastSection(True)
        self.tbl_stats.setEditTriggers(QTableWidget.NoEditTriggers)
        
        # Populate the table with empty rows for all gestures
        self.tbl_stats.setRowCount(len(FOLDER_TO_CLASS))
        self.gestures = list(FOLDER_TO_CLASS.keys())
        self.gesture_to_row = {}
        
        # Reset local metrics count
        self.reset_metrics()

        stats_layout.addWidget(self.tbl_stats)
        
        self.btn_export = QPushButton("Export Statistics (CSV)")
        self.btn_export.clicked.connect(self.export_csv_report)
        stats_layout.addWidget(self.btn_export)
        
        right_layout.addWidget(stats_group)

        # B. Run History logs
        history_group = QGroupBox("TEST HISTORY")
        history_layout = QVBoxLayout(history_group)
        self.lst_history = QListWidget()
        history_layout.addWidget(self.lst_history)
        
        btn_clear_hist = QPushButton("Clear History")
        btn_clear_hist.clicked.connect(self.lst_history.clear)
        history_layout.addWidget(btn_clear_hist)
        
        right_layout.addWidget(history_group)

        main_layout.addWidget(right_panel)

        self.append_log("[SYSTEM] Started successfully. GUI is ready.")

    def reset_metrics(self):
        self.metrics_data = {}
        for row, name in enumerate(self.gestures):
            self.gesture_to_row[name] = row
            self.metrics_data[name] = {"total": 0, "detected": 0, "correct": 0}
            
            # Create Table Items
            self.tbl_stats.setItem(row, 0, QTableWidgetItem(name))
            self.tbl_stats.setItem(row, 1, QTableWidgetItem("0"))
            self.tbl_stats.setItem(row, 2, QTableWidgetItem("0"))
            
            # Progress Bar for accuracy visual representation
            progress = QProgressBar()
            progress.setRange(0, 100)
            progress.setValue(0)
            progress.setFormat("0%")
            self.tbl_stats.setCellWidget(row, 3, progress)
            
        self.tbl_stats.resizeColumnsToContents()

    def update_model_ui(self):
        model_file = Path(self.model_path)
        if model_file.exists():
            self.lbl_model_name.setText(f"Model: {model_file.name}\n(Path is valid)")
            self.append_log(f"[INFO] Auto-detected model: {self.model_path}")
        else:
            self.lbl_model_name.setText("Model: best.pt (Not Found)")
            self.append_log("[WARN] best.pt not found in root folder. Select manually using 'Select Model File'.")

    # Slots / Callbacks
    def browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select YOLO Model file (.pt)", "", "Model PyTorch (*.pt)")
        if file_path:
            self.model_path = file_path
            self.update_model_ui()

    def browse_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video files (*.mp4 *.avi *.mkv)")
        if file_path:
            self.lbl_video_path.setText(file_path)
            self.source_tabs.setCurrentIndex(1)
            self.append_log(f"[INFO] Selected video file: {Path(file_path).name}")

    def browse_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if folder_path:
            self.lbl_folder_path.setText(folder_path)
            self.source_tabs.setCurrentIndex(2)
            self.append_log(f"[INFO] Selected dataset folder: {folder_path}")

    def on_slider_changed(self):
        self.lbl_conf_val.setText(f"{self.slider_conf.value() / 100:.2f}")
        self.lbl_iou_val.setText(f"{self.slider_iou.value() / 100:.2f}")

    def append_log(self, msg):
        self.txt_log.append(msg)
        self.txt_log.ensureCursorVisible()

    def start_detection(self):
        # 1. Check if model path is valid
        if not Path(self.model_path).exists():
            QMessageBox.critical(self, "Error", "YOLO model file (.pt) not found! Please select a model file first.")
            return

        # 2. Identify selected source type
        tab_idx = self.source_tabs.currentIndex()
        if tab_idx == 0:
            source_type = "webcam"
            source_val = self.cb_webcam_idx.currentText()
            self.lbl_hud_info.setText("Webcam Live")
        elif tab_idx == 1:
            source_type = "video"
            source_val = self.lbl_video_path.text()
            if source_val == "No file selected" or not Path(source_val).exists():
                QMessageBox.critical(self, "Error", "Please select a video file first!")
                return
            self.lbl_hud_info.setText(f"Video: {Path(source_val).name}")
        elif tab_idx == 2:
            source_type = "folder"
            source_val = self.lbl_folder_path.text()
            if not Path(source_val).exists():
                QMessageBox.critical(self, "Error", "Dataset folder does not exist! Please select again.")
                return
            self.lbl_hud_info.setText("Dataset Validation")

        # Reset UI statistics & graphics
        self.reset_metrics()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_pause.setEnabled(True)
        self.btn_pause.setText("Pause")
        self.btn_screenshot.setEnabled(True)

        conf_val = self.slider_conf.value() / 100.0
        iou_val = self.slider_iou.value() / 100.0

        # Start Worker Thread
        self.worker = VideoWorker(self.model_path, source_type, source_val, conf_val, iou_val)
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.stats_updated.connect(self.update_stats)
        self.worker.finished_folder.connect(self.folder_testing_finished)
        self.worker.log_message.connect(self.append_log)
        self.worker.start()

    def stop_detection(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self.worker = None

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.btn_screenshot.setEnabled(False)
        self.lbl_hud_fps.setText("FPS: --")
        self.lbl_view.setPixmap(QPixmap())
        self.lbl_view.setText("LIVE STREAM - STOPPED\n\nSelect a source and click 'START DETECTION' to resume.")
        self.append_log("[SYSTEM] Stopped model inference.")

    def toggle_pause(self):
        if self.worker:
            self.worker.paused = not self.worker.paused
            if self.worker.paused:
                self.btn_pause.setText("Resume")
                self.append_log("[SYSTEM] Paused video stream.")
            else:
                self.btn_pause.setText("Pause")
                self.append_log("[SYSTEM] Resumed video stream.")

    @Slot(QImage, dict)
    def update_frame(self, q_image, meta_info):
        self.current_frame = q_image.copy() # Cache the image for screenshots
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.lbl_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.lbl_view.setPixmap(scaled_pixmap)

        # Update FPS or progress counters
        if "fps" in meta_info and meta_info["fps"] > 0:
            self.lbl_hud_fps.setText(f"FPS: {meta_info['fps']:.1f}")
        
        if "index" in meta_info:
            self.lbl_hud_info.setText(f"Folder: {meta_info['current_folder']} | Image: {meta_info['index']}/{meta_info['total']}")
            self.lbl_hud_fps.setText(f"Reading: {meta_info['current_image']}")

    @Slot(str, bool, str)
    def update_stats(self, folder_name, is_detected, predicted_class):
        if folder_name not in self.metrics_data:
            return

        expected_class = FOLDER_TO_CLASS[folder_name]
        self.metrics_data[folder_name]["total"] += 1
        if is_detected:
            self.metrics_data[folder_name]["detected"] += 1
            if predicted_class == expected_class:
                self.metrics_data[folder_name]["correct"] += 1

        # Update row in QTableWidget
        row = self.gesture_to_row[folder_name]
        total = self.metrics_data[folder_name]["total"]
        correct = self.metrics_data[folder_name]["correct"]
        detected = self.metrics_data[folder_name]["detected"]
        
        acc = (correct / total) * 100 if total > 0 else 0.0

        self.tbl_stats.item(row, 1).setText(str(total))
        self.tbl_stats.item(row, 2).setText(str(detected))
        
        # Update progress bar
        progress = self.tbl_stats.cellWidget(row, 3)
        if isinstance(progress, QProgressBar):
            progress.setValue(int(acc))
            progress.setFormat(f"{acc:.1f}%")

    @Slot(dict)
    def folder_testing_finished(self, final_stats):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.btn_screenshot.setEnabled(False)

        # Calculate average accuracy
        total_images = sum(final_stats[c]["total"] for c in final_stats)
        correct_detections = sum(final_stats[c]["correct"] for c in final_stats)
        avg_acc = (correct_detections / total_images) * 100 if total_images > 0 else 0.0

        # Save to Run History List
        timestamp = datetime.datetime.now().strftime("%d/%m %H:%M")
        history_str = f"[{timestamp}] Folder - Acc: {avg_acc:.1f}% ({correct_detections}/{total_images} images)"
        self.lst_history.addItem(history_str)

        # Show final message box
        QMessageBox.information(
            self, 
            "Test Finished", 
            f"Completed dataset scan!\n\nTotal images: {total_images}\nCorrect detections: {correct_detections}\nAverage accuracy: {avg_acc:.1f}%"
        )
        self.append_log(f"[SUCCESS] Scanned dataset. Avg Accuracy: {avg_acc:.1f}%")

    def take_screenshot(self):
        if self.current_frame is None or self.current_frame.isNull():
            return
            
        # Create output directory for screenshots
        out_dir = Path(__file__).resolve().parent / "dist" / "screenshots"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = out_dir / f"screenshot_{timestamp}.jpg"
        
        # Save QImage
        success = self.current_frame.save(str(filename))
        if success:
            self.append_log(f"[SUCCESS] Saved screenshot at: {filename.name}")
        else:
            self.append_log("[ERR] Cannot save screenshot.")

    def export_csv_report(self):
        # Export metrics data to a CSV report
        out_dir = Path(__file__).resolve().parent / "dist" / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = out_dir / f"report_{timestamp}.csv"
        
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write("Class,Total,Detected,Correct,Accuracy%\n")
                total_t = 0
                total_d = 0
                total_c = 0
                for name, data in self.metrics_data.items():
                    acc = (data["correct"] / data["total"]) * 100 if data["total"] > 0 else 0
                    f.write(f"{name},{data['total']},{data['detected']},{data['correct']},{acc:.2f}\n")
                    total_t += data["total"]
                    total_d += data["detected"]
                    total_c += data["correct"]
                avg_acc = (total_c / total_t) * 100 if total_t > 0 else 0
                f.write(f"TOTAL,{total_t},{total_d},{total_c},{avg_acc:.2f}\n")
                
            self.append_log(f"[SUCCESS] Exported report file at: {filename.name}")
            QMessageBox.information(self, "Export Report", f"Report successfully exported at:\n{filename}")
        except Exception as e:
            self.append_log(f"[ERR] Error exporting report: {str(e)}")
            QMessageBox.critical(self, "Error", f"Cannot export report file: {str(e)}")

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
