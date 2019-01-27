import cv2


class BaseFrameHandler:
    """
    Base class for FrameHandler
    """
    def __init__(self):
        pass

    def get_symbols(self, frame):
        """
        Returns the list of symbol boxes found in a frame
        """
        raise Exception("Unimplemented error")

    def show_frames(self, frame, symbols, expressions):
        """
        Displays the symbols along with detected expressions on the given frame
        """
        for i, symbol in enumerate(symbols):
            cv2.rectangle(frame, (symbol.left, symbol.top), (symbol.right, symbol.bottom), (0, 255, 0), 3)
            cv2.putText(frame, symbol.prediction_cls, org=(symbol.left, symbol.top),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(0, 255, 0), thickness=2)

        for exp_box in expressions:
            cv2.rectangle(frame, (exp_box.left, exp_box.top), (exp_box.right, exp_box.bottom), (255, 0, 0), 1)
        cv2.imshow('detected symbols and expressions', frame)
