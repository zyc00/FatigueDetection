import dlib
import numpy as np
import cv2
import matplotlib.pyplot as plt

MODEL_PATH = "models/shape_predictor_70_face_landmarks.dat"


def detect_landmarks(image: np.ndarray, visualize=False):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(MODEL_PATH)

    left_eye_indices = [36, 37, 38, 39, 40, 41]
    right_eye_indices = [42, 43, 44, 45, 46, 47]
    mouse_indices = [49, 51, 52, 53, 55, 57, 58, 59]

    rect = detector(image, 0)[0]
    image = image[rect.top() : rect.bottom(), rect.left() : rect.right()]
    # rect_dlib = dlib.rectangle(rect.left(), rect.top(), rect.right(), rect.bottom())
    rect_dlib = dlib.rectangle(0, 0, image.shape[1], image.shape[0])

    points = []
    [points.append((p.x, p.y)) for p in predictor(image, rect_dlib).parts()]

    if visualize:
        plt.imshow(image[..., ::-1])
        [
            plt.text(p[0], p[1], str(i))
            for i, p in enumerate(points)
            if i in left_eye_indices or i in right_eye_indices or i in mouse_indices
        ]
        plt.savefig("landmarks.jpg")
        plt.close()

    return points


def check_fatigue(landmark: list[tuple[int, int]], image: np.ndarray, visualize=False):
    left_eye = np.array(landmark[36:42])
    right_eye = np.array(landmark[42:48])
    mouse = np.array(landmark[48:68])

    left_eye_width = left_eye[3][0] - left_eye[0][0]
    right_eye_width = right_eye[3][0] - right_eye[0][0]
    left_eye_height = left_eye[5][1] - left_eye[1][1]
    right_eye_height = right_eye[5][1] - right_eye[1][1]

    left_eye_aspect_ratio = left_eye_height / left_eye_width
    right_eye_aspect_ratio = right_eye_height / right_eye_width
    eye_aspect_ratio = (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2

    mouse_width = mouse[6][0] - mouse[0][0]
    mouse_height = mouse[9][1] - mouse[3][1]
    mouse_aspect_ratio = mouse_height / mouse_width

    left_eye_area = cv2.contourArea(left_eye)
    right_eye_area = cv2.contourArea(right_eye)
    eye_area = (left_eye_area + right_eye_area) / 2
    return (mouse_aspect_ratio > 0.7), (eye_area < 135)


def fatigue_detection(images: list[np.ndarray], visualize=False):
    fatigue_threshold = 0.5
    landmarks = [detect_landmarks(img, visualize) for img in images]
    eye_fatigue = [
        check_fatigue(landmark, img, visualize)[0]
        for landmark, img in zip(landmarks, images)
    ]
    mouth_fatigue = [
        check_fatigue(landmark, img, visualize)[1]
        for landmark, img in zip(landmarks, images)
    ]
    fatigue_score_eye = sum(eye_fatigue) / len(eye_fatigue)
    fatigue_score_mouth = sum(mouth_fatigue) / len(mouth_fatigue)
    if fatigue_score_eye > fatigue_threshold or fatigue_score_mouth > fatigue_threshold:
        return True
    return False
