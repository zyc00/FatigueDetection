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
    rect_dlib = dlib.rectangle(rect.left(), rect.top(), rect.right(), rect.bottom())

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

    left_eye_hull = cv2.convexHull(left_eye)
    right_eye_hull = cv2.convexHull(right_eye)
    mouse_hull = cv2.convexHull(mouse)

    cv2.drawContours(image, [left_eye_hull], -1, (0, 255, 0), 1)
    cv2.drawContours(image, [right_eye_hull], -1, (0, 255, 0), 1)
    cv2.drawContours(image, [mouse_hull], -1, (0, 255, 0), 1)

    left_eye_hull_area = cv2.contourArea(left_eye_hull)
    right_eye_hull_area = cv2.contourArea(right_eye_hull)
    mouse_hull_area = cv2.contourArea(mouse_hull)

    eye_aspect_ratio = (left_eye_hull_area + right_eye_hull_area) / mouse_hull_area

    if visualize:
        plt.imshow(image[..., ::-1])
        plt.savefig("fatigue.jpg")

    return eye_aspect_ratio


def fatigue_detection(images: list[np.ndarray], visualize=False):
    threshold = 0.3
    fatigue_threshold = 0.5
    landmarks = [detect_landmarks(img, visualize) for img in images]
    fatigue = [
        check_fatigue(landmark, img, visualize)
        for landmark, img in zip(landmarks, images)
    ]
    is_fatigue = [1 if f < threshold else 0 for f in fatigue]
    fatigue_score = sum(is_fatigue) / len(is_fatigue)
    if fatigue_score > fatigue_threshold:
        return True
    return False
