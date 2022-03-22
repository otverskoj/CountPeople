from collections import OrderedDict
from typing import List, Tuple
from scipy.spatial import distance
import numpy as np


class CentroidTracker:
    def __init__(self, max_disappeared: int = 50, max_distance: int = 50) -> None:
        ''' Инициализирует ID следующего уникального объекта
            вместе с двумя OrderedDict, используемыми для отслеживания
            сопоставления ID данного объекта с его центроидом
            и количеством последовательных кадров, после которых
            объект помечается как "исчезнувший". '''
        self.next_object_id = 0
        self.objects, self.disappeared = OrderedDict(), OrderedDict()
        self.max_distance = max_distance
        # кол-во последовательных кадров, по пошествию которых
        # данный объект может быть помечен как "исчезнувший"
        self.max_disappeared = max_disappeared
    
    def register(self, centroid: Tuple[float, float]) -> None:
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
    
    def deregister(self, object_id: int) -> None:
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, rects: List[Tuple[float, float, float, float]]) -> OrderedDict:
        if len(rects) == 0:
            for obj_id in self.disappeared.keys():
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects
        
        input_centroids = np.zeros((len(rects), 2), dtype='int')
        for i, start_x, start_y, end_x, end_y in enumerate(rects):
            centroid_x = (start_x + end_x) // 2.0
            centroid_y = (start_y + end_y) // 2.0
            input_centroids[i] = centroid_x, centroid_y
        
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = self.objects.keys()
            object_centroids = self.objects.values()

            dists = distance.cdist(object_centroids, input_centroids)

            rows = dists.min(axis=1).argsort()
            cols = dists.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                if dists[row, col] < self.max_distance:
                    obj_id = object_ids[row]
                    self.objects[obj_id] = input_centroids[col]
                    self.disappeared[obj_id] = 0

                    used_rows.add(row)
                    used_cols.add(col)
            
            unused_rows = set(dists.shape[0]).difference(used_rows)
            unused_cols = set(dists.shape[1]).difference(used_cols)

            if len(object_centroids) >= len(input_centroids):
                for row in unused_rows:
                    obj_id = object_ids[row]
                    self.disappeared[obj_id] += 1
                    if self.disappeared[obj_id] > self.max_disappeared:
                        self.deregister(obj_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])
            
        return self.objects
