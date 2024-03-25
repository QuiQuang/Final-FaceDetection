import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QLabel,
    QStackedWidget,
    QComboBox,
    QProgressBar,
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
import cv2


class HomePage(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Tạo combobox để chọn chức năng
        self.comboBox = QComboBox()
        self.comboBox.addItem("Home")
        self.comboBox.addItem("Camera")
        self.comboBox.addItem("Other Widget")
        layout.addWidget(self.comboBox)

        # Tạo label để chứa hình ảnh
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Tạo và cấu hình QStackedWidget để chứa các trang
    stacked_widget = QStackedWidget()

    # HomePage
    home_page = HomePage()
    stacked_widget.addWidget(home_page)
    stacked_widget.setWindowTitle("Cheating Detection")
    stacked_widget.setGeometry(100, 100, 640, 480)
    stacked_widget.show()

    sys.exit(app.exec_())
