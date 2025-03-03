import cv2
from fatigue.fat_det import fatigue_detection
from glob import glob

pos_img_files = glob("positive_images/*.jpg")
neg_img_files = glob("negative_images/*.jpg")
pos_img_files.sort()
neg_img_files.sort()

pos_imgs = [cv2.imread(img) for img in pos_img_files]
neg_imgs = [cv2.imread(img) for img in neg_img_files]

print(fatigue_detection(pos_imgs, visualize=True))
print(fatigue_detection(neg_imgs, visualize=True))
