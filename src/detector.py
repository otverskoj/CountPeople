import os
import numpy as np
import cv2


class Detector:
    def __init__(self,
                 model_type: str,
                 model_dir: str,
                 confidence: float = 0.9) -> None:
        self.model_type = model_type
        self.model_dir = model_dir
        self.confidence = confidence

        self.model_mapping = {
            'mobilenet_ssd': {
                'init': self.init_mobilenetssd,
                'detect': self.detect_by_mobilenetssd
            },
            'yolov3': {
                'init': self.init_yolo,
                'detect': self.detect_by_yolo
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

    def init_yolo(self):
        pass

    def detect(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        return self.model_mapping[self.model_type]['detect'](image)
    
    def detect_by_mobilenetssd(self, image: np.ndarray) -> np.ndarray:
        print(type(image))
        print(image.shape)
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843,
                                     (300, 300), 127.5)
        self.model.setInput(blob)
        detections = self.model.forward()

        result = []
        for i in np.arange(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence:
                class_idx = int(detections[0, 0, i, 1])
                bbox = detections[0, 0, i, 3:7] * np.array((w, h, w, h))
                bbox = bbox.astype('int')
                result.append(tuple(bbox, confidence, class_idx))
        
        return np.array(result)
    
    def detect_by_yolo(image):
        pass
    
    def display(self, image_path: str, data: np.ndarray) -> None:
        image = cv2.imread(image_path)
        
        for i in range(data.shape[0]):
            bbox, confidence, class_idx = data[i]

            label = f'{self.classes[class_idx]}: {confidence * 100:.2f}%'
            cv2.rectangle(image, bbox[:2], bbox[2:], (0, 255, 0), 2)
            text_y = bbox[1] - 15 if bbox[1] > 30 else bbox[1] + 15
            cv2.putText(image, label, (bbox[0], text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow("Output", image)
        cv2.waitKey(0)


if __name__ == '__main__':
    detector = Detector('mobilenet_ssd', './model_data/mobilenet_ssd')
    data = detector.detect('../data/input/image/frame.jpg')
    detector.display('../data/input/image/frame.jpg', data)
    