import os
import time

import cv2
import imutils
from imutils.video import FPS
import numpy as np
import dlib
import torch
from facenet_pytorch import MTCNN

from src.centroid_tracker import CentroidTracker
# from src.face_detector import FaceDetector
from src.detector import Detector
from src.trackable_object import TrackableObject


# TODO: исправить все TODO внутри функции
# TODO: добавить логику ROI
# TODO: добавить оптимизацию
def process_video(input_file: str, 
                  output_file: str,
                  skip_frames: int = 30, 
                  confidence: float = 0.4) -> None:
    if not os.path.isfile(input_file):
        print('There is no such file. Check path of input video')
        return
    
    video_capture = cv2.VideoCapture(input_file)

    OUT_VIDEO_W, OUT_VIDEO_H = 800, 600
    writer = cv2.VideoWriter(output_file, 
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             25, 
                             (OUT_VIDEO_W, OUT_VIDEO_H),
                             True)

    centroid_tracker = CentroidTracker(max_disappeared=40)
    trackers, trackable_objects = [], {}

    total_frames = 0
    total_in, total_out = 0, 0

    fps_counter = FPS().start()

    detector = Detector('yolov5', './model_data/yolov5', confidence_threshold=confidence)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
            
        status = 'Waiting'
        rects = []

        if total_frames % skip_frames == 0:
            status = 'Detecting'
            trackers = []

            # TODO: заменить детектор
            detections = detector.detect(frame)

            for detection in detections:
                if detector.classes[detection[2]] != 'person':
                    continue

                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(*tuple(map(int, detection[0])))
                tracker.start_track(frame, rect)

                trackers.append(tracker)
        else:
            status = 'Tracking'
            for tracker in trackers:
                tracker.update(frame)
                pos = tracker.get_position()

                coords = pos.left(), pos.top(), pos.right(), pos.bottom()
                rects.append(tuple(map(int, coords)))
        
        # линия для понимания, зашёл человек или вышел
        # TODO: изменить координаты точек линии
        cv2.line(frame, (850, 450), (1350, 450), (0, 255, 255), 2)
        LINE_H = 450

        objects = centroid_tracker.update(rects)

        for obj_id, centroid in objects.items():
            trackable_object = trackable_objects.get(obj_id, None)

            if trackable_object is None:
                trackable_object = TrackableObject(obj_id, centroid)
            else:
                # TODO: реализовать логику подсчёта in/out людей
                y = [c[1] for c in trackable_object.centroids]
                direction = centroid[1] - np.mean(y)

                trackable_object.centroids.append(centroid)

                if not trackable_object.is_counted:
                    if direction < 0 and centroid[1] < LINE_H:
                        total_in += 1
                        trackable_object.is_counted = True
                    elif direction > 0 and centroid[1] > LINE_H:
                        total_out += 1
                        trackable_object.is_counted = True
            
            trackable_objects[obj_id] = trackable_object

            text = f'ID {obj_id}'
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, centroid, 4, (0, 255, 0), -1)
        
        info = (
            f'In: {total_in}',
            f'Out: {total_out}',
            f'Status: {status}'
        )

        for i, text in enumerate(info):
            cv2.putText(frame, text, (10, OUT_VIDEO_H - ((i * 20) + 20)),
			            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        writer.write(frame)

        total_frames += 1
        fps_counter.update()
    
    fps_counter.stop()
    print("Elapsed time: {:.2f}".format(fps_counter.elapsed()))
    print("Approx. FPS: {:.2f}".format(fps_counter.fps()))

    writer.release()
    video_capture.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    process_video(input('Path to input video: ', 'Path to output video: '))
