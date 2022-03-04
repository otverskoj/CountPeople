try:
    from PIL import Image
    from facenet_pytorch import MTCNN, InceptionResnetV1
except ImportError:
    print('Libs are not Installed')
else:
    print('OK')