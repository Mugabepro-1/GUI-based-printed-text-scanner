import sys
import cv2
import pytesseract
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
    QTextEdit, QSplitter, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread
from PyQt6.QtGui import QImage, QPixmap


class CameraWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(0)
        # Kali fix: set buffer size + better backend
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while self.running:
            ret, frame = cap.read()
            if ret:
                self.frame_ready.emit(frame.copy())

        cap.release()

    def stop(self):
        self.running = False
        self.wait()


class TextScannerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Printed Text Scanner Pro - Kali Edition")
        self.setGeometry(100, 100, 1280, 720)
        self.setStyleSheet("background-color: #0d1117; color: #c9d1d9;")

        self.image = None
        self.original_image = None
        self.roi = None
        self.drawing = False
        self.start_point = None
        self.ocr_data = None

        self.init_ui()
        self.setup_camera()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # === Left: Image View ===
        self.image_label = QLabel("Drag an image here or start camera")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                background: #161b22;
                border: 3px dashed #30363d;
                border-radius: 12px;
                font-size: 18px;
                color: #8b949e;
            }
        """)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.mousePressEvent = self.mouse_press
        self.image_label.mouseMoveEvent = self.mouse_move
        self.image_label.mouseReleaseEvent = self.mouse_release

        # === Right: Controls & Text ===
        controls = QVBoxLayout()

        btns = QHBoxLayout()
        self.load_btn = QPushButton("Load Image")
        self.camera_btn = QPushButton("Start Camera")
        self.ocr_btn = QPushButton("Run OCR on ROI")

        for btn in (self.load_btn, self.camera_btn, self.ocr_btn):
            btn.setStyleSheet("""
                QPushButton {
                    padding: 12px;
                    font-size: 14px;
                    border-radius: 8px;
                    background: #238636;
                    color: white;
                    font-weight: bold;
                }
                QPushButton:hover { background: #2ea043; }
                QPushButton:pressed { background: #1a6f2a; }
            """)

        self.load_btn.clicked.connect(self.load_image)
        self.camera_btn.clicked.connect(self.toggle_camera)
        self.ocr_btn.clicked.connect(self.run_ocr)

        btns.addWidget(self.load_btn)
        btns.addWidget(self.camera_btn)
        btns.addWidget(self.ocr_btn)

        self.text_display = QTextEdit()
        self.text_display.setPlaceholderText("Extracted text will appear here...")
        self.text_display.setStyleSheet("background: #161b22; border-radius: 8px; padding: 10px;")

        controls.addLayout(btns)
        controls.addWidget(QLabel("<b><font color='#58a6ff'>Extracted Text:</font></b>"))
        controls.addWidget(self.text_display)

        # === Splitter ===
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.image_label)
        splitter.addWidget(QWidget())
        splitter.setSizes([850, 430])

        right_widget = QWidget()
        right_widget.setLayout(controls)
        splitter.replaceWidget(1, right_widget)

        layout.addWidget(splitter)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)")
        if path:
            self.image = cv2.imread(path)
            if self.image is None:
                QMessageBox.warning(self, "Error", "Cannot load image!")
                return
            self.original_image = self.image.copy()
            self.roi = None
            self.display_image(self.image)
            self.stop_camera()

    def display_image(self, cv_img):
        if cv_img is None:
            return

        display = cv_img.copy()

        # Draw ROI
        if self.roi:
            x, y, w, h = self.roi
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(display, "ROI", (x, y - 15), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)

        # Draw OCR overlay
        if self.ocr_data and 'text' in self.ocr_data:
            for i in range(len(self.ocr_data['text'])):
                conf = int(self.ocr_data['conf'][i])
                if conf > 30:
                    x = self.ocr_data['left'][i]
                    y = self.ocr_data['top'][i]
                    w = self.ocr_data['width'][i]
                    h = self.ocr_data['height'][i]
                    text = self.ocr_data['text'][i].strip()
                    cv2.rectangle(display, (x, y), (x + w, y + h), (255, 150, 50), 2)
                    cv2.putText(display, text, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 50), 2)

        # Convert BGR → RGB → QImage
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

    # === ROI Selection ===
    def mouse_press(self, event):
        if not self.image_label.pixmap():
            return
        self.drawing = True
        self.start_point = (event.pos().x(), event.pos().y())

    def mouse_move(self, event):
        if self.drawing and self.image_label.pixmap():
            temp = self.original_image.copy() if self.original_image is not None else self.image.copy()
            x1, y1 = self.start_point
            x2, y2 = event.pos().x(), event.pos().y()
            cv2.rectangle(temp, (x1, y1), (x2, y2), (0, 255, 0), 3)
            self.display_image(temp)

    def mouse_release(self, event):
        if not self.drawing or not self.image_label.pixmap():
            return
        self.drawing = False

        pixmap = self.image_label.pixmap()
        if not pixmap:
            return

        # Scale coordinates back to original image
        label_w = self.image_label.width()
        label_h = self.image_label.height()
        img_w = self.image.shape[1]
        img_h = self.image.shape[0]

        scale_x = img_w / label_w
        scale_y = img_h / label_h

        x1, y1 = self.start_point
        x2, y2 = event.pos().x(), event.pos().y()

        x = int(min(x1, x2) * scale_x)
        y = int(min(y1, y2) * scale_y)
        w = int(abs(x2 - x1) * scale_x)
        h = int(abs(y2 - y1) * scale_y)

        if w > 20 and h > 20:
            self.roi = (x, y, w, h)
            self.display_image(self.image)
            self.run_ocr()  # Auto OCR on ROI select

    def run_ocr(self):
        if self.image is None:
            self.text_display.setPlainText("No image loaded!")
            return

        target = self.image
        if self.roi:
            x, y, w, h = self.roi
            target = self.image[y:y + h, x:x + w]

        gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.convertScaleAbs(gray, alpha=2.0, beta=10)

        config = "--oem 3 --psm 6"
        try:
            self.ocr_data = pytesseract.image_to_data(
                enhanced, config=config, output_type=pytesseract.Output.DICT
            )
            full_text = pytesseract.image_to_string(enhanced, config=config)
            self.text_display.setPlainText(full_text.strip())
            self.display_image(self.image)  # refresh with overlay
        except Exception as e:
            QMessageBox.critical(self, "OCR Error", str(e))

    # === Camera Handling (Fixed for Kali) ===
    def setup_camera(self):
        self.camera_thread = None

    def toggle_camera(self):
        if self.camera_thread and self.camera_thread.isRunning():
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        self.camera_thread = CameraWorker()
        self.camera_thread.frame_ready.connect(self.update_camera_frame)
        self.camera_thread.start()
        self.camera_btn.setText("Stop Camera")
        self.camera_btn.setStyleSheet("background: #da3633;")

    def stop_camera(self):
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
        self.camera_btn.setText("Start Camera")
        self.camera_btn.setStyleSheet("")
        self.image = None
        self.roi = None
        self.image_label.clear()
        self.image_label.setText("Camera stopped")

    def update_camera_frame(self, frame):
        self.image = frame
        self.original_image = frame.copy()
        self.display_image(frame)
        # Optional live OCR if ROI is selected
        if self.roi:
            self.run_ocr()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Best dark theme on Kali
    window = TextScannerApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()