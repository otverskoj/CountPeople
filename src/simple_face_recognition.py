from facenet_pytorch import MTCNN
import torch
import numpy as np
import mmcv, cv2
from PIL import Image, ImageDraw


ROI = np.array((
    (250, 1440),
    (515, 888), 
    (850, 450),
    (1350, 450),
    (1700, 600),
    (2560, 600),
    (2560, 1440),
    (250, 1440)
))


def main(input_file: str, output_file: str):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    
    mtcnn = MTCNN(keep_all=True, device=device)
    
    video = mmcv.VideoReader(input_file)

    frames_tracked = []
    for i, frame in enumerate(video, start=1):

        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        print(f'\rTracking frame: {i}', end='')
        
        # Detect faces
        boxes, _ = mtcnn.detect(frame)
        
        # Draw faces
        frame_draw = frame.copy()
        drawer = ImageDraw.Draw(frame_draw)
        if boxes is not None:
            for box in boxes:
                # up_left, down_right = box[:2], box[2:]
                # if is_inside(up_left, ROI) and is_inside(down_right, ROI):
                #     draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
                drawer.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
        
        frames_tracked.append(frame_draw.resize((640, 360), Image.BILINEAR))
    print('\nDone')

    dim = frames_tracked[0].size
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')    
    video_tracked = cv2.VideoWriter(output_file, fourcc, 25.0, dim)
    for frame in frames_tracked:
        video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    video_tracked.release()


def is_inside(point: np.ndarray, polygon: np.ndarray) -> bool:
    in_polygon = False
    x, y = point
    for i in range(polygon.shape[0]):
        curr_x, curr_y = polygon[i]
        prev_x, prev_y = polygon[i - 1]
        if ((curr_y <= y and y < prev_y) or (prev_y <= y and y < curr_y)) and \
           (x > (prev_x - curr_x) * (y - curr_y) / (prev_y - curr_y) + curr_x):
                in_polygon = not in_polygon
    return in_polygon


if __name__ == '__main__':
    main('../data/input/video/clip.mp4', '../data/input/video/output.mp4')
