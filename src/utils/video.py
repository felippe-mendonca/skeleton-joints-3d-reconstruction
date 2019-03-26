import os
import cv2


class VideoIterator:
    def __init__(self, filename=None):

        self._filename = filename
        self._vc = cv2.VideoCapture(filename)
        if not self._vc.isOpened():
            raise Exception("Can't open video '{}'", filename)

        self._fps = self._vc.get(cv2.CAP_PROP_FPS)
        self._width = int(self._vc.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._n_frames = int(self._vc.get(cv2.CAP_PROP_FRAME_COUNT))

        self._begin = 0
        self._end = self._n_frames - 1

    def fps(self):
        return self._fps

    def resolution(self):
        return (self._width, self._height)
    
    def n_frames(self):
        return self._n_frames
    
    def in_range(self, iter_range):
        if iter_range.start >= self._n_frames:
            raise Exception("Start position of range is greater than number of frames.")

        self._begin = iter_range.start
        self._end = iter_range.stop

        self._vc = cv2.VideoCapture(self._filename)
        # drop initial frames out of range
        while True:
            next_frame_id = int(self._vc.get(cv2.CAP_PROP_POS_FRAMES))
            if next_frame_id == self._begin:
                break
            self._vc.read()

        return self

    def __iter__(self):
        return self

    def __next__(self):

        next_frame_id = int(self._vc.get(cv2.CAP_PROP_POS_FRAMES))
        if next_frame_id == self._n_frames or next_frame_id >= self._end:
            raise StopIteration()

        _, frame = self._vc.read()
        return frame
