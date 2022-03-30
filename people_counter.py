import os
from typing import Iterable, Tuple

import cv2
from imutils.video import FPS
import numpy as np
import dlib

from src.centroid_tracker import CentroidTracker
# from src.face_detector import FaceDetector
from src.detector import Detector
from src.trackable_object import TrackableObject


# TODO: исправить все TODO внутри функции
# TODO: добавить логику ROI
# TODO: добавить оптимизацию
def process_video(input_file: str = None, 
                  output_file: str = None,
                  to_screen: bool = False,
                  skip_frames: int = 25, 
                  confidence: float = 0.4) -> None:
    if not os.path.isfile(input_file):
        print('There is no such file. Check path of input video')
        return
    
    video_capture = cv2.VideoCapture(input_file)

    OUT_W, OUT_H = 800, 600
    OUT_W_SCALE, OUT_H_SCALE = None, None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_file, 
                             fourcc,
                             25, 
                             (OUT_H, OUT_W),
                             True)

    # CentroidTracker - это по сути маппер, который мапит 
    # какой-то признак каждого человека c ID, чтобы можно было вести подсчёт.
    # В данном случае признак - просто центроид (центр bbox'а).
    centroid_tracker = CentroidTracker(max_disappeared=40)
    trackers = []
    trackable_objects = {}  # "мапит" object_id и экземпляр TrackableObject

    total_frames = 0
    total_in, total_out = 0, 0

    fps_counter = FPS().start()

    detector = Detector('yolov5', './model_data/yolov5', confidence_threshold=confidence)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        if OUT_W_SCALE is None and OUT_H_SCALE is None:
            OUT_H_SCALE = OUT_H / frame.shape[0]
            OUT_W_SCALE = OUT_W / frame.shape[1]
        
        # уменьшение размера и размерности frame для увеличения скорости работы
        resized_rgb_frame = cv2.cvtColor(
            cv2.resize(frame, (OUT_W, OUT_H)),
            cv2.COLOR_BGR2RGB
        )
            
        status = 'Waiting'
        bboxes = []  # bbox'ы, возвращаемые либо detector'ом, либо correlation tracker'ами

        # запускаем процесс обнаружения, обновляем трэкеры
        if total_frames % skip_frames == 0:
            status = 'Detecting'
            detections = detect(resized_rgb_frame, detector)
            trackers = start_track(resized_rgb_frame, detections)
        else:
            status = 'Tracking'
            new_bboxes = update_trackers(trackers, resized_rgb_frame)
            bboxes.extend(new_bboxes)
        
        # линия для понимания, зашёл человек или вышел
        LINE_H = int(630 * OUT_H_SCALE)
        LINE_LEFT_W, LINE_RIGHT_W = int(875 * OUT_W_SCALE), int(1350 * OUT_W_SCALE)
        cv2.line(
            resized_rgb_frame,
            (LINE_LEFT_W, LINE_H), 
            (LINE_RIGHT_W, LINE_H),
            (0, 255, 0), 2
        )

        objects = centroid_tracker.update(bboxes)

        # считаем людей
        for obj_id, centroid in objects.items():
            trackable_object = trackable_objects.get(obj_id, None)

            if trackable_object is None:
                trackable_object = TrackableObject(obj_id, centroid)
            else:
                # TODO: реализовать логику подсчёта in/out людей
                
                # напрвление (вниз или вверх идёт человек)
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
            cv2.putText(resized_rgb_frame, text, (centroid[0] - 10, centroid[1] - 10),
			            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(resized_rgb_frame, centroid, 4, (0, 255, 0), -1)
        
        info = (
            f'In: {total_in}',
            f'Out: {total_out}',
            f'Status: {status}'
        )

        for i, text in enumerate(info):
            cv2.putText(resized_rgb_frame, text, (10, OUT_H - ((i * 20) + 20)),
			            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        resized_bgr_frame = cv2.cvtColor(resized_rgb_frame,
                                         cv2.COLOR_RGB2BGR)
        if to_screen:
            # if resized_rgb_frame.shape[0] > 1080 or resized_rgb_frame.shape[1] > 1920:
            #     resized_rgb_frame = cv2.resize(resized_rgb_frame, (1920, 1080))
        
            cv2.imshow("Detections", resized_bgr_frame)
            if cv2.waitKey(1) == ord("q"):
                break
        else:
            writer.write(resized_bgr_frame)

        total_frames += 1
        fps_counter.update()
    
    fps_counter.stop()
    print("Elapsed time: {:.2f}".format(fps_counter.elapsed()))
    print("Approx. FPS: {:.2f}".format(fps_counter.fps()))

    writer.release()
    video_capture.release()

    cv2.destroyAllWindows()


def detect(frame: np.ndarray, detector: Detector) -> Tuple:
    detections = detector.detect(image=frame)
    return tuple([
        d[0] for d in detections if detector.classes[d[2]] == 'person'
    ])


def start_track(frame: np.ndarray, detections: Iterable) -> Iterable:
    trackers = []
    for detection in detections:
        tracker = dlib.correlation_tracker()
        left, top, width, height = map(int, detection)
        rect = dlib.rectangle(left, top, left + width, top + height)
        tracker.start_track(frame, rect)
        trackers.append(tracker)
    return trackers


def update_trackers(trackers: Iterable, frame: np.ndarray) -> Iterable[Tuple]:
    result = []
    for tracker in trackers:
        tracker.update(frame)
        pos = tracker.get_position()
        coords = pos.left(), pos.top(), pos.right(), pos.bottom()
        result.append(tuple(map(int, coords)))
    return result


if __name__ == '__main__':
    process_video(r'D:\Work\CountPeople\data\input\video\clip.mp4',
                  r'D:\Work\CountPeople\data\output\video\clip.mp4',
                  True)
