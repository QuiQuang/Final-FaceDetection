import sys
import cv2
import numpy as np
import os
import mediapipe as mp
import time
import utils, add_function
import math
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QLabel,
    QStackedWidget,
    QTabWidget,
    QHBoxLayout,
    QMessageBox,
)
from PyQt5.QtCore import Qt, QTimer, QFile, QTextStream, QSize
from PyQt5.QtGui import QImage, QPixmap, QRadialGradient, QColor, QIcon
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import model_from_json

CLOSED_EYES_FRAME = 3
FONTS = cv2.FONT_HERSHEY_COMPLEX

FACE_OVAL = [
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
]
LIPS = [
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    308,
    324,
    318,
    402,
    317,
    14,
    87,
    178,
    88,
    95,
    185,
    40,
    39,
    37,
    0,
    267,
    269,
    270,
    409,
    415,
    310,
    311,
    312,
    13,
    82,
    81,
    42,
    183,
    78,
]
LOWER_LIPS = [
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    308,
    324,
    318,
    402,
    317,
    14,
    87,
    178,
    88,
    95,
]
UPPER_LIPS = [
    185,
    40,
    39,
    37,
    0,
    267,
    269,
    270,
    409,
    415,
    310,
    311,
    312,
    13,
    82,
    81,
    42,
    183,
    78,
]
LEFT_EYE = [
    362,
    382,
    381,
    380,
    374,
    373,
    390,
    249,
    263,
    466,
    388,
    387,
    386,
    385,
    384,
    398,
]
LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
RIGHT_EYE = [
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    173,
    157,
    158,
    159,
    160,
    161,
    246,
]
RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]


# Home page
class HomePage(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        label_home = QLabel("This is Home tab", self)
        label_home.setAlignment(Qt.AlignCenter)
        layout.addWidget(label_home)


# Camera page
class CameraPage(QWidget):
    def __init__(self):
        super().__init__()
        self.cap = None
        self.camera_started = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # Init
        self.camera_width = 1400
        self.camera_height = 900

        # Tạo container chứa camera
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
        info_layout_button = QVBoxLayout(self.info_container)
        layout.addWidget(self.info_container)
        layout.setStretch(1, 1)

        # # Tạo button để bắt đầu và dừng camera
        # icon = QIcon("image/camera-on.png")
        # self.control_button = QPushButton()
        # self.control_button.setIcon(icon)
        # info_layout_button.addWidget(self.control_button)
        # self.control_button.clicked.connect(self.toggle_camera)
        # self.control_button.setStyleSheet(
        #     "QPushButton {outline: 0; text-align: center; height: 34px; padding: 0 13px; vertical-align: top; border-radius: 3px; border: 2px solid transparent; background: #fff; border-color: #9B9B9B; color: #000; font-weight: 600; text-transform: uppercase; line-height: 16px; font-size: 11px;}\
        #     QPushButton:hover {background: #e8e8e8; color: #3d3d3d;}"
        # )
        # self.control_button.setFixedSize(250, 50)
        # self.control_button.setIconSize(QSize(50, 30))
        # info_layout_button.setAlignment(Qt.AlignCenter)

        # # Init model detection
        # self.face_cascade = cv2.CascadeClassifier(
        #     "model/haarcascade_frontalface_default.xml"
        # )

        # # Load Anti-Spoofing Model graph
        # json_file = open("model/build.json", "r")
        # loaded_model_json = json_file.read()
        # json_file.close()
        # self.model = model_from_json(loaded_model_json)
        # self.model.load_weights("model_weights/model-6.h5")

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
        black_image = np.zeros(
            (self.camera_height, self.camera_width, 3), dtype=np.uint8
        )
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
                if preds > 0.5:
                    label = "SPOOF"
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
                    label = "REAL"
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
                QPixmap.fromImage(image).scaled(self.camera_width, self.camera_height)
            )


# Eye keyboard page
class EyeKeyboardPage(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        label_other = QLabel("This is Eye Keyboard tab", self)
        label_other.setAlignment(Qt.AlignCenter)
        layout.addWidget(label_other)


# Cheat deection page
class CheatDetectionPage(QWidget):
    def __init__(self):
        super().__init__()
        self.cap = None
        self.camera_started = False
        self.timer1 = QTimer(self)
        self.timer1.timeout.connect(self.update_frame)

        self.timer2 = QTimer(self)
        self.timer2.timeout.connect(self.show_popup)
        self.timer2.setInterval(5000)

        # Init
        self.camera_height = 930
        self.camera_width = 1100
        self.popup_shown = False

        # Init model
        self.model = mp.solutions.face_mesh
        self.cef_counter = 0
        self.total_blinks = 0
        self.eye_position = "CENTER"
        self.eye_position_left = "CENTER"
        self.face_direct = "CENTER"
        self.ratio = 0

        # Tạo container chứa camera và thông tin
        layout = QHBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setStretch(0, 8)

        # Tạo label để hiển thị hình ảnh từ camera
        self.camera_label = QLabel()
        layout.addWidget(self.camera_label)

        # Tạo container blank ở giữa
        self.blank = QWidget()
        self.blank.setFixedWidth(15)
        layout.addWidget(self.blank)
        layout.setStretch(1, 0)

        # Tạo container chứa thông tin và button
        self.info_container = QWidget()
        info_layout_button = QVBoxLayout(self.info_container)
        layout.addWidget(self.info_container)
        layout.setStretch(2, 1)

        # Tạo container chứa thông tin xuất ra
        self.info_label = QLabel("<b>SHOW</b>")
        self.info_label_ratio = QLabel(f"Ratio: {self.ratio}")
        self.info_label_blink = QLabel(f"Total blinks: {self.total_blinks}")
        self.info_label_lepos = QLabel(f"Left eye: {self.eye_position_left}")
        self.info_label_repos = QLabel(f"Right eye: {self.eye_position}")
        self.info_label_face = QLabel(f"Face: {self.face_direct}")

        self.icon_label = QLabel()
        pixmap = QPixmap("image/logo.png").scaled(100, 100)
        self.icon_label.setPixmap(pixmap)
        self.icon_label.setFixedSize(100, 100)

        info_layout_button.addWidget(self.info_label)
        self.info_label.setAlignment(Qt.AlignCenter)
        info_layout_button.addWidget(self.info_label_ratio)
        info_layout_button.addWidget(self.info_label_blink)
        info_layout_button.addWidget(self.info_label_lepos)
        info_layout_button.addWidget(self.info_label_repos)
        info_layout_button.addWidget(self.info_label_face)
        info_layout_button.addWidget(self.icon_label, alignment=Qt.AlignCenter)

        self.info_label.setStyleSheet("font-weight: bold; font-size: 50px;")
        self.info_label_ratio.setStyleSheet("font-size: 25px;font-weight: bold")
        self.info_label_blink.setStyleSheet("font-size: 25px;font-weight: bold")
        self.info_label_lepos.setStyleSheet("font-size: 25px;font-weight: bold")
        self.info_label_repos.setStyleSheet("font-size: 25px;font-weight: bold")
        self.info_label_face.setStyleSheet("font-size: 25px;font-weight: bold")
        self.show_black_camera()

        # Tạo button để bắt đầu và dừng camera
        icon = QIcon("image/camera-on.png")
        self.control_button = QPushButton()
        self.control_button.setIcon(icon)
        info_layout_button.addWidget(
            self.control_button, alignment=Qt.AlignBottom | Qt.AlignCenter
        )
        self.control_button.clicked.connect(self.toggle_camera)
        self.control_button.setStyleSheet(
            "QPushButton {outline: 0; text-align: center; height: 34px; padding: 0 13px; vertical-align: top; border-radius: 3px; border: 2px solid transparent; background: #fff; border-color: #9B9B9B; color: #000; font-weight: 600; text-transform: uppercase; line-height: 16px; font-size: 11px;} QPushButton:hover {background: #e8e8e8; color: #3d3d3d;}"
        )
        self.control_button.setFixedSize(250, 50)
        self.control_button.setIconSize(QSize(50, 30))

    def reset_var(self):
        self.total_blinks = 0
        self.eye_position = ""
        self.eye_position_left = ""
        self.face_direct = ""

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
            self.timer1.start(30)
            self.timer2.start()
            self.camera_started = True
            icon = QIcon("image/camera-off.png")
            self.control_button.setIcon(icon)

    def stop_camera(self):
        # Dừng timer nếu đang hoạt động
        if self.timer1.isActive():
            self.timer1.stop()
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
        black_image = np.zeros(
            (self.camera_height, self.camera_width, 3), dtype=np.uint8
        )
        black_image.fill(0)
        height, width, channel = black_image.shape
        bytes_per_line = 3 * width
        q_img = QImage(
            black_image.data, width, height, bytes_per_line, QImage.Format_RGB888
        )
        self.camera_label.setPixmap(QPixmap.fromImage(q_img))
        self.info_label_ratio.setText("Ratio:")
        self.info_label_blink.setText("Total blinks:")
        self.info_label_lepos.setText("Left eye:")
        self.info_label_repos.setText("Right eye:")
        self.info_label_face.setText("Face:")

    def update_frame(self):
        # Đọc frame từ camera
        ret, frame = self.cap.read()

        if ret:
            with self.model.FaceMesh(
                min_detection_confidence=0.5, min_tracking_confidence=0.5
            ) as face_mesh:
                frame = cv2.resize(
                    frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC
                )
                frame_height, frame_width = frame.shape[:2]

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                results = face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    mesh_coords = add_function.landmarksDetection(frame, results, False)

                    # Tính toán ratio và vẽ lên hình
                    self.ratio = add_function.blinkRatio(
                        frame, mesh_coords, RIGHT_EYE, LEFT_EYE
                    )
                    utils.colorBackgroundText(
                        frame,
                        f"Ratio : {round(self.ratio,2)}",
                        FONTS,
                        0.7,
                        (30, 100),
                        2,
                        utils.PINK,
                        utils.YELLOW,
                    )

                    # Tính toán lên blink
                    if self.ratio > 4.8:
                        self.cef_counter += 1
                        utils.colorBackgroundText(
                            frame,
                            f"Blink",
                            FONTS,
                            1.7,
                            (int(frame_height / 2), 100),
                            2,
                            utils.YELLOW,
                            pad_x=6,
                            pad_y=6,
                        )
                    else:
                        if self.cef_counter > CLOSED_EYES_FRAME:
                            self.total_blinks += 1
                            self.cef_counter = 0

                    utils.colorBackgroundText(
                        frame,
                        f"Total Blinks: {self.total_blinks}",
                        FONTS,
                        0.7,
                        (30, 150),
                        2,
                    )

                    # Lấy thông tin xử lí
                    top_head = mesh_coords[10]
                    bot_chin = mesh_coords[152]
                    face_coords = [mesh_coords[p] for p in FACE_OVAL]
                    right_coords = [mesh_coords[p] for p in RIGHT_EYE]
                    left_coords = [mesh_coords[p] for p in LEFT_EYE]
                    eye_coords = [mesh_coords[p] for p in LEFT_EYE + RIGHT_EYE]

                    # Xử lí tính tủng tâm của 2 mắt
                    eye_center = (
                        round(sum(coord[0] for coord in eye_coords) / len(eye_coords)),
                        round(sum(coord[1] for coord in eye_coords) / len(eye_coords)),
                    )
                    face_center = (
                        round(
                            sum(coord[0] for coord in face_coords) / len(face_coords)
                        ),
                        round(
                            sum(coord[1] for coord in face_coords) / len(face_coords)
                        ),
                    )

                    # Tính toán xử lí mặt nghiêng
                    slope_detect_face = add_function.calulate_slope(top_head, bot_chin)
                    angle_detect_face = add_function.calculate_angle(slope_detect_face)
                    new_size = add_function.calculate_new_size(frame, angle_detect_face)

                    # Tính toán các giá trị cần lấy ra sau khi mặt nghiêng
                    face_center_after = add_function.rotate_point(
                        face_center,
                        angle_detect_face,
                        (new_size[0] // 2, new_size[1] // 2),
                        new_size,
                        (frame.shape[1], frame.shape[0]),
                    )
                    eye_center_after = add_function.rotate_point(
                        eye_center,
                        angle_detect_face,
                        (new_size[0] // 2, new_size[1] // 2),
                        new_size,
                        (frame.shape[1], frame.shape[0]),
                    )
                    self.face_direct = add_function.get_direct_face(
                        face_center_after, eye_center_after
                    )

                    # Cắt ảnh mắt để xử lí
                    crop_right, crop_left = add_function.eyesExtractor(
                        frame, right_coords, left_coords
                    )

                    self.eye_position, color = add_function.positionEstimator(
                        crop_right
                    )
                    utils.colorBackgroundText(
                        frame,
                        f"R: {self.eye_position}",
                        FONTS,
                        1.0,
                        (40, 220),
                        2,
                        color[0],
                        color[1],
                        8,
                        8,
                    )
                    self.eye_position_left, color = add_function.positionEstimator(
                        crop_left
                    )
                    utils.colorBackgroundText(
                        frame,
                        f"L: {self.eye_position_left}",
                        FONTS,
                        1.0,
                        (40, 320),
                        2,
                        color[0],
                        color[1],
                        8,
                        8,
                    )
                    self.update_info()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(
            frame.data,
            frame.shape[1],
            frame.shape[0],
            QImage.Format_RGB888,
        )
        self.camera_label.setPixmap(
            QPixmap.fromImage(image).scaled(self.camera_width, self.camera_height)
        )

    def update_info(self):
        self.info_label_ratio.setText(f"Ratio: {round(self.ratio, 2)}")
        self.info_label_blink.setText(f"Total blinks: {self.total_blinks}")
        self.info_label_lepos.setText(f"Left eye: {self.eye_position_left}")
        self.info_label_repos.setText(f"Right eye: {self.eye_position}")
        self.info_label_face.setText(f"Face: {self.face_direct}")

    def reset_timer2(self):
        self.timer2.stop()
        self.popup_shown = False
        self.timer2.start()

    def show_popup(self):
        if not self.popup_shown and (
            self.face_direct != "CENTER"
            or (self.eye_position != "CENTER" and self.eye_position_left != "CENTER")
        ):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Warning")
            msg.setText("Please pay attention to the screen!")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            self.popup_shown = True
            self.reset_timer2()


# Main page
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


# Responsive tab
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
    widget.setWindowTitle("App - Sept")
    widget.setWindowIcon(QIcon("image/logo.png"))

    # Setting
    widget.show()
    sys.exit(app.exec_())
