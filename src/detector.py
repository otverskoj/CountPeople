import os
import time
from typing import Tuple
import numpy as np
import cv2


class Detector:
    def __init__(self,
                 model_type: str,
                 model_dir: str,
                 confidence_threshold: float = 0.9,
                 nms_threshold: float = 0.45) -> None:
        self.model_type = model_type
        self.model_dir = model_dir
        self.confidence_threshold = confidence_threshold
        self.nms_thershold = nms_threshold

        self.model_mapping = {
            'mobilenet_ssd': {
                'init': self.init_mobilenetssd,
                'detect': self.detect_by_mobilenetssd
            },
            'yolov5': {
                'init': self.init_yolov5,
                'detect': self.detect_by_yolov5
            }
        }
        self.model_mapping[model_type]['init']()

    def init_mobilenetssd(self) -> None:
        filenames = list(os.walk(self.model_dir))[0][2]
        for filename in filenames:
            tmp = os.path.join(self.model_dir, filename)
            if filename.endswith('.txt'):
                classes_file = tmp        
            elif filename.endswith('.prototxt'):
                prototxt_file = tmp
            elif filename.endswith('.caffemodel'):
                caffe_file = tmp

        with open(classes_file, mode='r', encoding='utf-8') as f:
            self.classes = tuple(c.strip() for c in f.readlines())
        self.model = cv2.dnn.readNetFromCaffe(prototxt_file, caffe_file)

    def init_yolov5(self):
        filenames = list(os.walk(self.model_dir))[0][2]
        for filename in filenames:
            tmp = os.path.join(self.model_dir, filename)
            if filename.endswith('.txt'):
                classes_file = tmp
            elif filename.endswith('.onnx'):
                onnx_file = tmp

        with open(classes_file, mode='r', encoding='utf-8') as f:
            self.classes = tuple(c.strip() for c in f.readlines())
        self.model = cv2.dnn.readNetFromONNX(onnx_file)

    def detect(self, image_path: str = None, 
               image: np.ndarray = None) -> Tuple:
        if image_path is not None:
            image = cv2.imread(image_path)
        
        return self.model_mapping[self.model_type]['detect'](image)
    
    def detect_by_mobilenetssd(self, image: np.ndarray) -> Tuple:
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843,
                                     (300, 300), 127.5)
        self.model.setInput(blob)
        detections = self.model.forward()

        result = []
        for i in np.arange(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                class_idx = int(detections[0, 0, i, 1])
                bbox = detections[0, 0, i, 3:7] * np.array((w, h, w, h))
                bbox = bbox.astype('int')
                result.append(tuple([bbox, confidence, class_idx]))
        
        return np.array(result)
    
    def detect_by_yolov5(self, image: np.ndarray) -> Tuple:
        yolov5_dim = 640

        height, width = image.shape[:2]

        blob = cv2.dnn.blobFromImage(cv2.resize(image, (yolov5_dim, yolov5_dim)), 
                                     1 / 255.0, (yolov5_dim, yolov5_dim))        
        self.model.setInput(blob)
        predictions = self.model.forward()
        output = predictions[0]

        rows = output.shape[0]

        x_factor = width / yolov5_dim
        y_factor =  height / yolov5_dim

        bboxes, confidences, class_ids = [], [], []
        for row_idx in range(rows):
            row = output[row_idx]
            confidence = row[4]
            if confidence > self.confidence_threshold:
                classes_scores = row[5:]
                _, _, _, max_idx = cv2.minMaxLoc(classes_scores)
                class_id = max_idx[1]
                if classes_scores[class_id] > 0.25:
                    x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                    left = int((x - 0.5 * w) * x_factor)
                    top = int((y - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    bbox = np.array([left, top, width, height])
                    
                    bboxes.append(bbox)
                    confidences.append(confidence)
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(bboxes, confidences, 
                                   self.confidence_threshold, 
                                   self.nms_thershold)

        result = tuple(
            tuple([bboxes[i], confidences[i], class_ids[i]]) for i in indices
        )
        
        return result

    def display(self, image_path: str = None, 
                image: np.ndarray = None,
                with_draw: bool = False,
                data: Tuple = tuple()) -> None:
        if image_path is not None:
            image = cv2.imread(image_path)
        
        if with_draw:
            image = self.draw_bboxes(image, data)
        
        if image.shape[0] > 1080 or image.shape[1] > 1920:
            image = cv2.resize(image, (1920, 1080))
        
        cv2.imshow("Detections", image)
        if cv2.waitKey(1) == ord('q'):
            return
    
    def draw_bboxes(self, image: np.ndarray, data: Tuple) -> np.ndarray:
        for detection in data:
            bbox, confidence, class_idx = detection
            color = (0, 255, 0)
            
            # # mobilenet_ssd display
            # label = f'{self.classes[class_idx]}: {confidence * 100:.2f}%'
            # cv2.rectangle(image, bbox[:2], bbox[2:], color, 2)
            # text_y = bbox[1] - 15 if bbox[1] > 30 else bbox[1] + 15
            # cv2.putText(image, label, (bbox[0], text_y), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # yolo display
            cv2.rectangle(image, bbox, color, 2)
            cv2.putText(image, f'{self.classes[class_idx]}: {confidence * 100:.2f}%', 
                        (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 0, 0))
        
        return image
    
    def detect_video(self, video_path: str = None,
                     output_path: str = None) -> None:
        video = cv2.VideoCapture(0 if video_path is None else video_path)
        
        if not video.isOpened():
            print("Cannot open camera")
            return
        
        if video_path is None:
            time.sleep(2.0)
        
        writer = None
        output_w, output_h = None, None

        while True:
            ret, frame = video.read()

            if video_path is not None and not ret:
                print("Can't receive frame (stream end?). Exiting...")
                break

            # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame = frame

            data = self.detect(image=rgb_frame)
            rgb_frame = self.draw_bboxes(rgb_frame, data)

            if output_w is None and output_h is None:
                # TODO: можно сделать другое разрешение
                output_w, output_h = 800, 600
            
            if output_path is not None and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                writer = cv2.VideoWriter(output_path, fourcc,
                                         25, (output_h, output_w), True)
            
            writer.write(rgb_frame)
            
            # if video_path is None or output_path is None:
            #     self.display(image=rgb_frame)
        
        video.release()
        writer.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    detector = Detector('yolov5', './model_data/yolov5', confidence_threshold=0.4)
    # image_path = './data/input/image/frame.jpg'
    # data = detector.detect(image_path)
    # detector.display(image_path, data)
    detector.detect_video(r'D:\Work\CountPeople\data\input\video\clip.mp4', 
                          r'D:\Work\CountPeople\data\output\video\clip.mp4')
