import cv2

cap = cv2.VideoCapture(
    "/home/jigu/projects/FatigueDetection/data/YawDD dataset/Dash/Female/1-FemaleNoGlasses.avi"
)
i = 0
while cap.isOpened():
    i = i + 1
    ret, frame = cap.read()
    if i in [1340, 1370, 1400, 1430]:
        cv2.imwrite(f"positive_images/{i}.jpg", frame)
    if i > 1440:
        break

cap.release()
