import sys
import os
import json
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel,
    QSpinBox, QCheckBox, QPushButton, QHBoxLayout, QDoubleSpinBox
)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')


class Configurator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Konfigurator")
        self.setFixedSize(300, 300)

        self.layout = QVBoxLayout()
        self._load_config()

        self.layout.addWidget(QLabel("Kadrowanie (x, y, szer, wys):"))
        self.crop_x = QSpinBox()
        self.crop_y = QSpinBox()
        self.crop_w = QSpinBox()
        self.crop_h = QSpinBox()
        for box in [self.crop_x, self.crop_y, self.crop_w, self.crop_h]:
            box.setRange(0, 1920)
        self.crop_x.setValue(self.cfg.get("frame_crop", {}).get("x", 0))
        self.crop_y.setValue(self.cfg.get("frame_crop", {}).get("y", 0))
        self.crop_w.setValue(self.cfg.get("frame_crop", {}).get("width", 0))
        self.crop_h.setValue(self.cfg.get("frame_crop", {}).get("height", 0))
        row = QHBoxLayout()
        for widget in [self.crop_x, self.crop_y, self.crop_w, self.crop_h]:
            row.addWidget(widget)
        self.layout.addLayout(row)

        self.layout.addWidget(QLabel("Confidence threshold:"))
        self.confidence = QDoubleSpinBox()
        self.confidence.setRange(0.0001, 1.0000)
        self.confidence.setDecimals(4)
        self.confidence.setSingleStep(0.0001)
        default_conf = self.cfg.get("detection", {}).get("confidence_threshold", 0.9991)
        self.confidence.setValue(float(default_conf))
        self.layout.addWidget(self.confidence)

        self.rect_cb = QCheckBox("Ramka")
        self.alert_cb = QCheckBox("Alert na ekranie")
        self.sound_cb = QCheckBox("Dźwięk")

        self.rect_cb.setChecked(self.cfg.get("alert", {}).get("show_rectangle", False))
        self.alert_cb.setChecked(self.cfg.get("alert", {}).get("show_alert", False))
        self.sound_cb.setChecked(self.cfg.get("alert", {}).get("play_sound", False))

        self.layout.addWidget(self.rect_cb)
        self.layout.addWidget(self.alert_cb)
        self.layout.addWidget(self.sound_cb)

        self.lines_cb = QCheckBox("Linie pomocnicze")
        self.lines_cb.setChecked(self.cfg.get("overlay", {}).get("show_lines", False))
        self.layout.addWidget(self.lines_cb)

        self.save_btn = QPushButton("Zapisz")
        self.save_btn.clicked.connect(self.save_config)
        self.layout.addWidget(self.save_btn)

        self.setLayout(self.layout)

    def _load_config(self):
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, 'r') as f:
                    self.cfg = json.load(f)
            except Exception:
                self.cfg = {}
        else:
            self.cfg = {}

    def save_config(self):
        config = {
            "frame_crop": {
                "x": self.crop_x.value(),
                "y": self.crop_y.value(),
                "width": self.crop_w.value(),
                "height": self.crop_h.value()
            },
            "detection": {
                "confidence_threshold": round(self.confidence.value(), 4)
            },
            "alert": {
                "show_rectangle": self.rect_cb.isChecked(),
                "show_alert": self.alert_cb.isChecked(),
                "play_sound": self.sound_cb.isChecked()
            },
            "overlay": {
                "show_lines": self.lines_cb.isChecked()
            }
        }
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=4)
        self.close()

    @staticmethod
    def load_user_config():
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config.json'))
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Could not load user config: {e}")
        return {}


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Configurator()
    window.show()
    sys.exit(app.exec())
