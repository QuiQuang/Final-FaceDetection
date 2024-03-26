import sys
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
from PyQt5.QtCore import Qt, QTimer, QFile, QTextStream
from PyQt5.QtGui import QImage, QPixmap, QRadialGradient, QColor
import cv2
import numpy as np

# Global
camera_width = 1200
camera_height = 900

button_style = "\
                    {QPushButton {position: relative; font-size: 17px; text-transform: uppercase; text-decoration: none; padding: 1em 2.5em; display: inline-block; border-radius: 6em; transition: all .2s; border: none; font-family: inherit; font-weight: 500; color: black; background-color: white;}\
                    QPushButton:hover {transform: translateY(-3px); box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);}\
                    QPusgButton:active {transform: translateY(-1px); box-shadow: 0 5px 10px rgba(0, 0, 0, 0.2);}\
                    QPushButton:after {display: inline-block; height: 100%; width: 100%; border-radius: 100px; position: absolute; top: 0; left: 0; z-index: -1; transition: all .4s; background-color: #fff;}\
                    QPushButton:hover::after {transform: scaleX(1.4) scaleY(1.6); opacity: 0;}\
                "


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
        self.control_button = QPushButton("START")
        info_layout.addWidget(self.control_button)
        self.control_button.clicked.connect(self.toggle_camera)
        self.control_button.setStyleSheet(button_style)
        self.control_button.setFixedSize(100, 50)
        info_layout.setAlignment(Qt.AlignCenter)

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
            self.control_button.setText("STOP")

    def stop_camera(self):
        # Dừng timer nếu đang hoạt động
        if self.timer.isActive():
            self.timer.stop()
        # Đóng camera
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        # Return
        self.camera_started = False
        self.control_button.setText("START")
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
            # Chuyển đổi frame sang định dạng RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Chuyển đổi frame thành QImage
            image = QImage(
                frame_rgb.data,
                frame_rgb.shape[1],
                frame_rgb.shape[0],
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
