from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout,
                             QGroupBox, QLabel, QCheckBox)
from PyQt5.QtCore import QPropertyAnimation, QEasingCurve
import sys

class CollapsibleBox(QGroupBox):
    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.toggle_button = QCheckBox(title)
        self.toggle_button.setChecked(True)
        self.content_area = QWidget()
        self.content_layout = QVBoxLayout(self.content_area)
        self.layout_main = QVBoxLayout(self)
        self.layout_main.addWidget(self.toggle_button)
        self.layout_main.addWidget(self.content_area)
        self.animation = QPropertyAnimation(self, b"maximumHeight")
        self.animation.setDuration(250)
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)
        self.toggle_button.stateChanged.connect(self.toggle_content)
        self.toggle_content(self.toggle_button.checkState())
        
    def toggle_content(self, state):
        self.animation.stop()
        collapsed_height = self.sizeHint().height() - self.content_area.maximumHeight()
        end_value = collapsed_height if state == 0 else self.sizeHint().height()
        self.animation.setStartValue(self.height())
        self.animation.setEndValue(end_value)
        self.animation.start()

    def add_widget(self, widget):
        self.content_layout.addWidget(widget)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QWidget()
    layout_window = QVBoxLayout(window)
    collapsible_box1 = CollapsibleBox("Section 1")
    collapsible_box1.add_widget(QLabel("Content for Section 1"))
    collapsible_box1.add_widget(QPushButton("Button 1"))
    collapsible_box2 = CollapsibleBox("Section 2")
    collapsible_box2.add_widget(QLabel("Content for Section 2"))
    collapsible_box2.add_widget(QPushButton("Button 2"))
    layout_window.addWidget(collapsible_box1)
    layout_window.addWidget(collapsible_box2)
    window.setWindowTitle("Collapsible Sections Example")
    window.show()
    sys.exit(app.exec_())