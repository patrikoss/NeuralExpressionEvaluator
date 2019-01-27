import cv2
from src.utils.base_frame_handler import BaseFrameHandler
from src.sliding_window.slider import SlidingWindow


class SlidingWindowFrameHandler(BaseFrameHandler):
    """
    Base class for FrameHandler
    """
    def __init__(self, detector):
        self.detector = detector

    def get_symbols(self, frame):
        """
        Returns the list of symbol boxes found in a frame
        """
        sw = SlidingWindow(frame, 2, 2, 0.5, 0.5, 3, 32, 32, self.detector)
        return sw.slide()

    def show_frames(self, frame, symbols, expressions):
        """
        Displays the symbols along with detected expressions on the given frame
        """
        super().show_frames(frame, symbols, expressions)
