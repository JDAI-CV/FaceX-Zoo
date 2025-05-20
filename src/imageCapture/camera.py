import cv2
import os
from datetime import datetime
from src.faceDetection.detection import FaceDetector
import subprocess
import platform

class CameraHandler:
    _SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'model')
    _HELP_TEXT = [
        "Make photo - 's'",
        "Quit - 'q'",
        "Begin analysis - 'a'",
        "Open configurator - 'c'"
    ]
    WINDOW_NAME = "Preview"
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720

    def __init__(self, camera_index=0):
        os.makedirs(self._SAVE_DIR, exist_ok=True)
        self._analyze_mode = False
        self._face_detector = None
        self.cap = cv2.VideoCapture(camera_index)
        self._running = True
        self.current_frame = None

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera.")

    def start(self):
        while self._running:
            success, frame = self.cap.read()
            self.current_frame = frame

            if not success:
                print("Error while fetching frame.")
                break

            if self._analyze_mode and self._face_detector:
                frame = self._face_detector.analyze(frame)

            self._show_helper_texts(frame)
            cv2.imshow(self.WINDOW_NAME, frame)
            self._handle_key()

        self._cleanup()

    def _handle_key(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self._running = False
        elif key == ord('s'):
            self._save_frame()
        elif key == ord('a'):
            self._begin_analysis()
        elif key == ord('c'):
            self._launch_configurator()

    def _cleanup(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

    def _show_helper_texts(self, frame):
        for i, line in enumerate(self._HELP_TEXT):
            cv2.putText(frame, line, (10, 25 + i * 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    def _is_opened(self):
        return self.cap.isOpened()

    def _save_frame(self):
        # Pobierz nową, czystą klatkę z kamery (bez helperów)
        ret, raw_frame = self.cap.read()
        if not ret:
            print("No frame to save.")
            return

        filename = os.path.join(
            self._SAVE_DIR,
            f"wzorzec_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        )

        try:
            is_saved = cv2.imwrite(filename, raw_frame)
            if not is_saved:
                raise ValueError(f"Cannot save file {filename}")
            print(f"Saved file to {filename}")
        except Exception as e:
            print(f"Saving error: {e}")

    def _begin_analysis(self):
        if self._analyze_mode:
            self._analyze_mode = False
            print("Analysis mode: OFF")
        else:
            self._face_detector = FaceDetector()
            self._analyze_mode = True
            print("Analysis mode: ON")

    def _launch_configurator(self):
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configurator', 'configurator.py'))
        print(f"Opening configurator: {config_path}")
        try:
            if platform.system() == "Windows":
                subprocess.Popen(['python', config_path], shell=True)
            else:
                subprocess.Popen(['python3', config_path])
        except Exception as e:
            print(f"Failed to launch configurator: {e}")

if __name__ == "__main__":
    camera = CameraHandler()
    camera.start()