from random import randint
import sys
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QWidget, QApplication, QVBoxLayout, QProgressBar

StyleSheet = '''
#RedProgressBar {
    text-align: center;
}
#RedProgressBar::chunk {
    background-color: #F44336;
}
#GreenProgressBar {
    min-height: 12px;
    max-height: 12px;
    border-radius: 6px;
}
#GreenProgressBar::chunk {
    border-radius: 6px;
    background-color: #009688;
}
#BlueProgressBar {
    border: 2px solid #2196F3;
    border-radius: 5px;
    background-color: #E0E0E0;
}
#BlueProgressBar::chunk {
    background-color: #2196F3;
    width: 10px; 
    margin: 0.5px;
}
'''


class ProgressBar(QProgressBar):

    def __init__(self, *args, **kwargs):
        super(ProgressBar, self).__init__(*args, **kwargs)
        self.setValue(0)
        if self.minimum() != self.maximum():
            self.timer = QTimer(self, timeout=self.onTimeout)
            self.timer.start(randint(1, 3) * 1000)

    def onTimeout(self):
        if self.value() >= 100:
            self.timer.stop()
            self.timer.deleteLater()
            del self.timer
            return
        self.setValue(self.value() + 1)


class Window(QWidget):

    def __init__(self, *args, **kwargs):
        super(Window, self).__init__(*args, **kwargs)
        self.resize(800, 600)
        layout = QVBoxLayout(self)
        layout.addWidget(
            ProgressBar(self, minimum=0, maximum=100, objectName="RedProgressBar"))

        layout.addWidget(
            ProgressBar(self, minimum=0, maximum=0, objectName="RedProgressBar"))

        layout.addWidget(
            ProgressBar(self, minimum=0, maximum=100, textVisible=False,
                        objectName="GreenProgressBar"))
        layout.addWidget(
            ProgressBar(self, minimum=0, maximum=0, textVisible=False,
                        objectName="GreenProgressBar"))

        layout.addWidget(
            ProgressBar(self, minimum=0, maximum=100, textVisible=False,
                        objectName="BlueProgressBar"))
        layout.addWidget(
            ProgressBar(self, minimum=0, maximum=0, textVisible=False,
                        objectName="BlueProgressBar"))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(StyleSheet)
    w = Window()
    w.show()
    sys.exit(app.exec_())