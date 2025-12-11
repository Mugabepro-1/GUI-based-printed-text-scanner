#!/usr/bin/env python3
import sys
import time
import cv2
import pytesseract
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
    QTextEdit, QSplitter, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap, QMouseEvent


class CameraWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, device_index=0):
        super().__init__()
        self.running = False
        self.device_index = device_index

    def run(self):
        self.running = True
        cap = cv2.VideoCapture(self.device_index, cv2.CAP_ANY)
        # small buffer and better backend hints
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if not cap.isOpened():
            self.running = False
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                # avoid busy loop if camera fails intermittently
                time.sleep(0.01)
                continue
            # emit a copy to avoid sharing mutable frames across threads
            self.frame_ready.emit(frame.copy())
            # small sleep to avoid maxed CPU
            time.sleep(0.01)

        cap.release()

    def stop(self):
        self.running = False
        # wait for thread to finish gracefully
        self.wait()


class TextScannerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Printed Text Scanner Pro - Fixed")
        self.setGeometry(100, 100, 1280, 720)
        self.setStyleSheet("background-color: #0d1117; color: #c9d1d9;")

        # main image data (cv2 BGR)
        self.image = None             # current image (BGR) used for OCR and ROI extraction
        self.original_image = None    # same as image copy for drawing purposes
        self.roi = None               # (x, y, w, h) in image coordinates
        self.drawing = False
        self.start_point_label = None  # start point in label coords while drawing
        self.ocr_data = None

        self.camera_thread = None

        self.init_ui()

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
        # override mouse handlers
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

    # ---------- Image loading & display ----------
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)")
        if not path:
            return

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.warning(self, "Error", "Cannot load image!")
            return

        # If camera is running, stop it but don't clear the image
        self.stop_camera(clear_image=False)

        self.image = img
        self.original_image = img.copy()
        self.roi = None
        self.ocr_data = None
        self.text_display.clear()
        self.display_image(self.image)

    def display_image(self, cv_img):
        """
        Display the provided BGR image on the QLabel keeping aspect ratio.
        Also draws ROI and OCR overlays if available.
        """
        if cv_img is None:
            return

        display = cv_img.copy()

        # Draw OCR overlay (coordinates from pytesseract are relative to the target image)
        if self.ocr_data and 'text' in self.ocr_data:
            n = len(self.ocr_data['text'])
            for i in range(n):
                text = self.ocr_data['text'][i].strip()
                try:
                    conf = float(self.ocr_data['conf'][i])
                except Exception:
                    # sometimes conf is '-1' as string
                    try:
                        conf = int(self.ocr_data['conf'][i])
                    except Exception:
                        conf = -1
                if text and conf > 30:
                    x = int(self.ocr_data['left'][i])
                    y = int(self.ocr_data['top'][i])
                    w = int(self.ocr_data['width'][i])
                    h = int(self.ocr_data['height'][i])
                    cv2.rectangle(display, (x, y), (x + w, y + h), (255, 150, 50), 2)
                    cv2.putText(display, text, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 50), 2)

        # Draw ROI
        if self.roi:
            x, y, w, h = self.roi
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(display, "ROI", (x, max(y-10,0)), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 2)

        # Convert BGR -> RGB -> QImage
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # Scale pixmap to label while keeping aspect ratio
        lbl_size = self.image_label.size()
        scaled = pixmap.scaled(lbl_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.image_label.setPixmap(scaled)

    # ---------- Helpers to map coordinates ----------
    def _pixmap_display_geometry(self):
        """
        Returns (offset_x, offset_y, pm_w, pm_h, scale) describing how the image pixmap is placed inside the label.
        - offset_x, offset_y: top-left position inside the label (label coordinates)
        - pm_w, pm_h: displayed pixmap size
        - scale: scale factor applied to original image to get displayed pixmap (float)
        """
        pixmap = self.image_label.pixmap()
        if pixmap is None or self.image is None:
            return None

        lbl_w = self.image_label.width()
        lbl_h = self.image_label.height()

        img_h, img_w = self.image.shape[0], self.image.shape[1]
        # compute same scale factor as used when scaling the pixmap in display_image
        scale = min(lbl_w / img_w, lbl_h / img_h)
        pm_w = int(img_w * scale)
        pm_h = int(img_h * scale)
        offset_x = (lbl_w - pm_w) // 2
        offset_y = (lbl_h - pm_h) // 2
        return offset_x, offset_y, pm_w, pm_h, scale

    def label_to_image_coords(self, lx, ly):
        """
        Convert coordinates relative to the label widget to coordinates in the original image.
        Returns (img_x, img_y) or None if outside the displayed pixmap.
        """
        geom = self._pixmap_display_geometry()
        if geom is None:
            return None
        offset_x, offset_y, pm_w, pm_h, scale = geom
        # check if point is inside the pixmap area
        if lx < offset_x or lx > offset_x + pm_w or ly < offset_y or ly > offset_y + pm_h:
            return None
        rel_x = lx - offset_x
        rel_y = ly - offset_y
        img_x = int(rel_x / scale)
        img_y = int(rel_y / scale)
        # clamp
        img_x = max(0, min(self.image.shape[1] - 1, img_x))
        img_y = max(0, min(self.image.shape[0] - 1, img_y))
        return img_x, img_y

    # ---------- Mouse handlers for ROI selection ----------
    def mouse_press(self, event: QMouseEvent):
        # Only start drawing if there's a displayed pixmap
        if not self.image_label.pixmap():
            return
        self.drawing = True
        # store starting mouse coords in label-space
        p = event.pos()
        self.start_point_label = (p.x(), p.y())

    def mouse_move(self, event: QMouseEvent):
        if not self.drawing or self.start_point_label is None:
            return
        if not self.image_label.pixmap():
            return

        # current label coords
        cur = (event.pos().x(), event.pos().y())

        # convert both to image coords; if either is outside pixmap, just ignore drawing
        start_img = self.label_to_image_coords(*self.start_point_label)
        cur_img = self.label_to_image_coords(*cur)
        if start_img is None or cur_img is None:
            # show the original image without the temporary rectangle
            self.display_image(self.image)
            return

        x1, y1 = start_img
        x2, y2 = cur_img

        temp = self.original_image.copy() if self.original_image is not None else self.image.copy()
        cv2.rectangle(temp, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # draw small label showing "Selecting..."
        cv2.putText(temp, "Selecting...", (min(x1, x2), max(10, min(y1, y2) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        self.display_image(temp)

    def mouse_release(self, event: QMouseEvent):
        if not self.drawing or self.start_point_label is None:
            return
        self.drawing = False

        if not self.image_label.pixmap():
            return

        end_point_label = (event.pos().x(), event.pos().y())
        start_img = self.label_to_image_coords(*self.start_point_label)
        end_img = self.label_to_image_coords(*end_point_label)
        self.start_point_label = None

        if start_img is None or end_img is None:
            # released outside the pixmap; ignore
            self.display_image(self.image)
            return

        x1, y1 = start_img
        x2, y2 = end_img

        x = int(min(x1, x2))
        y = int(min(y1, y2))
        w = int(abs(x2 - x1))
        h = int(abs(y2 - y1))

        if w > 10 and h > 10:
            self.roi = (x, y, w, h)
            # refresh with ROI drawn
            self.display_image(self.image)
            # auto-run OCR on ROI selection
            self.run_ocr()
        else:
            # too small; ignore
            self.roi = None
            self.display_image(self.image)

    # ---------- OCR ----------
    def run_ocr(self):
        if self.image is None:
            self.text_display.setPlainText("No image loaded!")
            return

        target = self.image
        if self.roi:
            x, y, w, h = self.roi
            # guard ROI bounds
            x = max(0, min(self.image.shape[1] - 1, x))
            y = max(0, min(self.image.shape[0] - 1, y))
            w = max(1, min(self.image.shape[1] - x, w))
            h = max(1, min(self.image.shape[0] - y, h))
            target = self.image[y:y + h, x:x + w]

        gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        # simple enhancement
        enhanced = cv2.convertScaleAbs(gray, alpha=2.0, beta=10)

        config = "--oem 3 --psm 6"
        try:
            self.ocr_data = pytesseract.image_to_data(
                enhanced, config=config, output_type=pytesseract.Output.DICT
            )
            full_text = pytesseract.image_to_string(enhanced, config=config)
            self.text_display.setPlainText(full_text.strip())
            # If OCR coords were for ROI, they are relative to the ROI image.
            # To draw them correctly on the full image, we must offset them with roi.x, roi.y
            if self.roi and self.ocr_data and 'left' in self.ocr_data:
                # offset the coordinates so overlay appears in the right place
                x_off, y_off = self.roi[0], self.roi[1]
                # make a deep copy and adjust coordinates (we keep the original in memory)
                od = dict(self.ocr_data)
                # lists need to be copied
                od['left'] = [int(l + x_off) for l in od['left']]
                od['top'] = [int(t + y_off) for t in od['top']]
                # widths/heights remain same
                self.ocr_data = od
            self.display_image(self.image)  # refresh with overlay
        except Exception as e:
            QMessageBox.critical(self, "OCR Error", str(e))

    # ---------- Camera handling ----------
    def setup_camera(self):
        # not used separately but kept for parity
        self.camera_thread = None

    def toggle_camera(self):
        if self.camera_thread and self.camera_thread.isRunning():
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        if self.camera_thread and self.camera_thread.isRunning():
            return
        self.camera_thread = CameraWorker(device_index=0)
        self.camera_thread.frame_ready.connect(self.update_camera_frame)
        self.camera_thread.start()
        self.camera_btn.setText("Stop Camera")
        self.camera_btn.setStyleSheet("background: #da3633;")
        # when camera starts, we want to show the live frames; clear ROI/ocr
        self.roi = None
        self.ocr_data = None
        self.text_display.clear()

    def stop_camera(self, clear_image=True):
        if self.camera_thread:
            try:
                self.camera_thread.stop()
            except Exception:
                pass
            self.camera_thread = None

        self.camera_btn.setText("Start Camera")
        self.camera_btn.setStyleSheet("")

        if clear_image:
            self.image = None
            self.roi = None
            self.original_image = None
            self.image_label.clear()
            self.image_label.setText("Camera stopped")

    def update_camera_frame(self, frame):
        # frame is BGR
        self.image = frame
        self.original_image = frame.copy()
        self.display_image(frame)
        # live OCR optionally when there is ROI
        if self.roi:
            # we avoid too-frequent OCR by not running it every single frame;
            # for simplicity here, we run it anyway (but in real app you'd throttle)
            self.run_ocr()

    # ---------- cleanup ----------
    def closeEvent(self, event):
        # ensure camera thread is stopped when window closes
        self.stop_camera(clear_image=False)
        event.accept()


def main():
    app = QApplication(sys.argv)
    # optional: set style
    try:
        app.setStyle("Fusion")
    except Exception:
        pass
    window = TextScannerApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
