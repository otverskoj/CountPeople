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
        self.model = cv2.dnn.readNet(onnx_file)

    def detect(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        return self.model_mapping[self.model_type]['detect'](image)
    
    def detect_by_mobilenetssd(self, image: np.ndarray) -> np.ndarray:
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
                result.append(tuple([bbox, confidence, class_idx]))
        
        return np.array(result)
    
    def detect_by_yolov5(self, image: np.ndarray) -> np.ndarray:
        yolov5_dim = 640

        height, weight = image.shape[:2]
        # _max = max(height, weight)
        # resized = np.zeros((_max, _max, 3), np.uint8)
        # resized[0:height, 0:weight] = image

        blob = cv2.dnn.blobFromImage(cv2.resize(image, (yolov5_dim, yolov5_dim)), 
                                     1 / 255.0, (yolov5_dim, yolov5_dim), swapRB=True)        
        self.model.setInput(blob)
        predictions = self.model.forward()
        output = predictions[0]

        rows = output.shape[0]

        x_factor = weight / yolov5_dim
        y_factor =  height / yolov5_dim

        bboxes, confidences, class_ids = [], [], []
        for r in range(rows):
            row = output[r]
            confidence = row[4]
            if confidence > self.confidence:
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

        indices = cv2.dnn.NMSBoxes(bboxes, confidences, 0.25, 0.45)

        result = []
        for i in indices:
            result.append(
                tuple([bboxes[i], confidences[i], class_ids[i]])
            )

        print(*result, sep='\n')
        print(bboxes, confidences, class_ids)

        return np.array(result)

    def display(self, image_path: str, data: np.ndarray) -> None:
        image = cv2.imread(image_path)
        
        for i in range(data.shape[0]):
            bbox, confidence, class_idx = data[i]
            color = (0, 255, 0)

            # label = f'{self.classes[class_idx]}: {confidence * 100:.2f}%'
            # cv2.rectangle(image, bbox[:2], bbox[2:], color, 2)
            # text_y = bbox[1] - 15 if bbox[1] > 30 else bbox[1] + 15
            # cv2.putText(image, label, (bbox[0], text_y), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            cv2.rectangle(image, bbox, color, 2)
            cv2.putText(image, f'{self.classes[class_idx]}: {confidence * 100:.2f}%', 
                        (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 0, 0))
        
        if image.shape[0] > 1080 or image.shape[1] > 1920:
            image = cv2.resize(image, (1920, 1080))
        
        cv2.imshow("Output", image)
        cv2.waitKey(0)


if __name__ == '__main__':
    detector = Detector('yolov5', './model_data/yolov5', 0.4)
    image_path = './data/input/image/frame.jpg'
    data = detector.detect(image_path)
    detector.display(image_path, data)
    