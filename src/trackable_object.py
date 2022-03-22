from typing import Tuple


class TrackableObject:
    def __init__(self, object_id: int, centroid: Tuple[float, float]) -> None:
        self.object_id = object_id
        self.centroids = [centroid]
        self.is_counted = False
