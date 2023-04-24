import cv2
import numpy as np
import os


def smooth_folder(src_folder, tgt_folder):
    image_files = os.listdir(src_folder)
    for image_file in image_files:
        
        src_file = os.path.join(src_folder, image_file)
        tgt_file = os.path.join(tgt_folder, image_file)

        img = cv2.imread(src_file)
        
        img = img.astype(np.float32)
        H, W, _ = img.shape
        M = 3
        img = cv2.resize(img, (int(H * M), int(W * M)))
        for i in range(15):
            img = cv2.GaussianBlur(img, (31, 31), cv2.BORDER_DEFAULT)

        cv2.imwrite(tgt_file, img)
