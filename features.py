import numpy as np
import cv2

# TODO handle nan values of rotation angle
def words_orientation(text_only_img, name=""):
    # Copying the image
    img = text_only_img.copy()

    # Horizontal projection
    hor_proj = np.sum(img, axis=1)
    baseline = np.argmin(hor_proj)

    # Flood fill algorithm to extract the words
    rotations = []
    h, w = img.shape
    for x in range(w-1):
        if img[baseline, x] == 255 and img[baseline, x+1] == 0:
            mask = np.zeros((h+2, w+2), np.uint8)
            cv2.floodFill(img, mask, (x+1, baseline), 255) # the mask is used to obtain the word
            mask *= 255
            mask = 255 - mask
            contours,_ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours: # it should be 1 contour only (1 word) but sometimes it is not
                rect = cv2.minAreaRect(cnt)
                _,(x,y),rotation = rect
                if(x*y < 0.7*h*w and x*y > 0.05*h*w): # discarding too big and too small contours
                    rotations.append(rotation)

    # returns the average of the words rotations
    return np.nanmean(rotations) 