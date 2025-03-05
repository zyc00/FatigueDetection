# import cv2
# from fatigue.fat_det import fatigue_detection
# from glob import glob

# pos_img_files = glob("positive_images/*.jpg")
# neg_img_files = glob("negative_images/*.jpg")
# pos_img_files.sort()
# neg_img_files.sort()

# pos_imgs = [cv2.imread(img) for img in pos_img_files]
# neg_imgs = [cv2.imread(img) for img in neg_img_files]
import numpy as np
from fatigue.fat_det import fatigue_detection
import cv2

images = np.load("close_eye.npy")
images = [image for image in images]
# for image in images:
#     cv2.imshow("image", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

print(fatigue_detection(images, visualize=True))
# print(fatigue_detection(neg_imgs, visualize=True))
