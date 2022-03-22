from typing import List, Tuple
import torch
import numpy as np
from facenet_pytorch import MTCNN


class FaceDetector:
    def __init__(self, roi: np.ndarray = None) -> None:
        self.roi = roi or np.array((
            (250, 1440),
            (515, 888), 
            (850, 450),
            (1350, 450),
            (1700, 600),
            (2560, 600),
            (2560, 1440),
            (250, 1440)
        ))
        self.source_shape = 2560, 1440
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(keep_all=True, device=self.device)

    def detect(self, frame: np.ndarray) -> Tuple[List[Tuple[float, float, float, float]], List[float]]:
        self._scale_roi(*frame.shape)
        boxes, probs = self.mtcnn.detect(frame)
        
        if boxes is None:
            return [], []
        
        bbox_in_roi, probs_for_bbox_in_roi = [], []
        for box, prob in zip(boxes, probs):
            left_up, right_down = box[:2], box[2:]
            if self._is_inside(left_up, self.roi) and \
                self._is_inside(right_down, self.roi):
                    bbox_in_roi.append(box)
                    probs_for_bbox_in_roi.append(prob)
        
        return tuple(bbox_in_roi), tuple(probs_for_bbox_in_roi)
    
    
    def _scale_roi(self, w: int, h: int) -> None:
        if self.source_shape == (w, h):
            return
        scale_factors = w / self.source_shape[0], h / self.source_shape[1]
        self.roi = self.roi * np.array(scale_factors)
        self.source_shape = w, h
    
    
    def _is_inside(point: np.ndarray, polygon: np.ndarray) -> bool:
        in_polygon = False
        x, y = point
        for i in range(polygon.shape[0]):
            curr_x, curr_y = polygon[i]
            prev_x, prev_y = polygon[i - 1]
            if ((curr_y <= y and y < prev_y) or (prev_y <= y and y < curr_y)) and \
            (x > (prev_x - curr_x) * (y - curr_y) / (prev_y - curr_y) + curr_x):
                    in_polygon = not in_polygon
        return in_polygon