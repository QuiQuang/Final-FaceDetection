import sys
import cv2
import numpy as np
import os

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QLabel,
    QStackedWidget,
    QTabWidget,
    QSpacerItem,
    QHBoxLayout,
    QSizePolicy,
)
from PyQt5.QtCore import Qt, QTimer, QFile, QTextStream, QSize
from PyQt5.QtGui import QImage, QPixmap, QRadialGradient, QColor, QIcon

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import model_from_json

# Global
camera_width = 1200
camera_height = 900


class HomePage(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        label_home = QLabel("This is Home tab", self)
        label_home.setAlignment(Qt.AlignCenter)
        layout.addWidget(label_home)


class CameraPage(QWidget):
    def __init__(self):
        super().__init__()
        self.cap = None
        self.camera_started = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # Tạo container chứa camera và thông tin
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 0)

        # Tạo label để hiển thị hình ảnh từ camera
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.camera_label)
        self.show_black_camera()
        layout.setStretch(0, 3)

        # Tạo container chứa thông tin cần thiết
        self.info_container = QWidget()
        info_layout = QVBoxLayout(self.info_container)
        layout.addWidget(self.info_container)
        layout.setStretch(1, 1)

        # Tạo button để bắt đầu và dừng camera
        icon = QIcon("image/camera-on.png")
        self.control_button = QPushButton()
        self.control_button.setIcon(icon)
        info_layout.addWidget(self.control_button)
        self.control_button.clicked.connect(self.toggle_camera)
        self.control_button.setStyleSheet(
            "QPushButton {display: inline-block; outline: 0; text-align: center; cursor: pointer; height: 34px; padding: 0 13px; vertical-align: top; border-radius: 3px; border: 2px solid transparent; transition: all .3s ease; background: #fff; border-color: #9B9B9B; color: #000; font-weight: 600; text-transform: uppercase; line-height: 16px; font-size: 11px;}\
            QPushButton:hover {background: #e8e8e8; color: #3d3d3d;}"
        )
        self.control_button.setFixedSize(100, 50)
        self.control_button.setIconSize(QSize(50, 30))
        info_layout.setAlignment(Qt.AlignCenter)

        # Init model detection
        self.face_cascade = cv2.CascadeClassifier(
            "./model/haarcascade_frontalface_default.xml"
        )

        # Load Anti-Spoofing Model graph
        json_file = open("./model/build.json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights("./model_weights/model-6.h5")

        print("Model loaded from disk")

    def toggle_camera(self):
        if not self.camera_started:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        # Kiểm tra xem camera đã được mở chưa
        if self.cap is None or not self.cap.isOpened():
            # Mở camera
            self.cap = cv2.VideoCapture(0)  # Số 0 là chỉ định sử dụng camera mặc định
            # Khởi động timer để cập nhật hình ảnh từ camera
            self.timer.start(30)
            # Return
            self.camera_started = True
            icon = QIcon("image/camera-off.png")
            self.control_button.setIcon(icon)

    def stop_camera(self):
        # Dừng timer nếu đang hoạt động
        if self.timer.isActive():
            self.timer.stop()
        # Đóng camera
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        # Return
        self.camera_started = False
        icon = QIcon("image/camera-on.png")
        self.control_button.setIcon(icon)
        self.show_black_camera()

    def show_black_camera(self):
        # Hiển thị màn hình đen
        black_image = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)
        black_image.fill(0)
        height, width, channel = black_image.shape
        bytes_per_line = 3 * width
        q_img = QImage(
            black_image.data, width, height, bytes_per_line, QImage.Format_RGB888
        )
        self.camera_label.setPixmap(QPixmap.fromImage(q_img))

    def update_frame(self):
        # Đọc frame từ camera
        ret, frame = self.cap.read()

        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for x, y, w, h in faces:
                face = frame[y - 5 : y + h + 5, x - 5 : x + w + 5]
                resized_face = cv2.resize(face, (160, 160))
                resized_face = resized_face.astype("float") / 255.0
                resized_face = img_to_array(resized_face)
                resized_face = np.expand_dims(resized_face, axis=0)
                preds = self.model.predict(resized_face)[0]
                print(preds)
                if preds > 0.5:
                    label = "Spoof"
                    cv2.putText(
                        frame,
                        label,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                else:
                    label = "Real"
                    cv2.putText(
                        frame,
                        label,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Chuyển đổi frame thành QImage
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(
                frame.data,
                frame.shape[1],
                frame.shape[0],
                QImage.Format_RGB888,
            )
            # Hiển thị hình ảnh trên QLabel
            self.camera_label.setPixmap(
                QPixmap.fromImage(image).scaled(camera_width, camera_height)
            )


class EyeKeyboardPage(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        label_other = QLabel("This is Eye Keyboard tab", self)
        label_other.setAlignment(Qt.AlignCenter)
        layout.addWidget(label_other)


class CheatDetectionPage(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        label_cheat_detection = QLabel("This is Cheat Detection tab", self)
        label_cheat_detection.setAlignment(Qt.AlignCenter)
        layout.addWidget(label_cheat_detection)


class Page(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Tạo QTabWidget để chứa các tab nhỏ
        tab_widget = ResponsiveTabWidget()
        layout.addWidget(tab_widget, stretch=1)

        # Tab "Home"
        home_tab = HomePage()
        tab_widget.addTab(home_tab, "Home".upper())

        # Tab "Detect Real or Spoof"
        camera_tab = CameraPage()
        tab_widget.addTab(camera_tab, "Real and Spoof".upper())

        # Tab "Eye tracking keyboard"
        other_widget_tab = EyeKeyboardPage()
        tab_widget.addTab(other_widget_tab, "Eye Keyboard".upper())

        # Tab "Cheating detection"
        cheat_detection_tab = CheatDetectionPage()
        tab_widget.addTab(cheat_detection_tab, "Cheat Detection".upper())


class ResponsiveTabWidget(QTabWidget):
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.adjustTabWidth()

    def adjustTabWidth(self):
        tab_bar = self.tabBar()
        total_width = self.width()
        num_tabs = self.count()
        if num_tabs > 0:
            tab_width = total_width / num_tabs
            for i in range(num_tabs):
                tab_bar.setStyleSheet(
                    f"QTabBar::tab {{ width: {tab_width}px; height: 80px; font: bold 24px; font-family: Calibri; }}"
                )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    # Tạo và cấu hình QStackedWidget để chứa các trang
    widget = QStackedWidget()

    page = Page()
    widget.addWidget(page)
    widget.setWindowTitle("Detection App")

    # Setting
    widget.show()
    sys.exit(app.exec_())
