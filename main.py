import mediapipe as mp
from nose_resize import NoseResizeTool

IMAGE_FILES = [
    'Resources/man1.jpg',
    'Resources/man2.jpg',
    'Resources/woman1.jpg',
    'Resources/woman2.jpg'
]
if __name__ == '__main__':
    mp_face_detecation = mp.solutions.face_detection
    face_detecation = mp_face_detecation.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    )
    nrt = NoseResizeTool(IMAGE_FILES[0],face_detecation)
    nrt.apply(0.8)
    nrt.show_result()