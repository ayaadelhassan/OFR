
# from skimage.feature import structure_tensor

# import skimage.io as io
import cv2

# gray_image = io.imread('stroke.png')
# output = structure_tensor(gray_image, sigma=0.1, order='rc')
# print(type(output))


# # tan2t = (2*Axy)/(Axx-Ayy)
# # t=np.arctan(tan2t)/2
# # data = t.ravel()


# # find the mode of `data` array (can use histogram). it is the direction.



from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
import os
import numpy as np



base_dir='ACdata_base/'

fonts = os.listdir(base_dir)
heights,widths=0,0
hMed,wMed=[],[]
total_len=0
for font in fonts:
    h,w = 0,0
    data = os.listdir(base_dir+font)
    print("curFont",font)
    for img in data:
        img_dir = base_dir+font+'/'+img
        imgGray = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
        hImg,wImg = imgGray.shape
        heights+=hImg
        h+=hImg
        widths+=wImg
        w+=wImg
        hMed.append(hImg)
        wMed.append(wImg)
    total_len+=len(data)
    print("mean of class: ",h/len(data),w/len(data))

print("mean of h,w :",heights/total_len,widths/total_len)
print("median of h,w :",np.median(hMed),np.median(wMed))



# img = imread('0210.jpg')
# plt.axis("off")
# plt.imshow(img)
# print(img.shape)

# resized_img = resize(img, (128*4, 64*4))
# plt.axis("off")
# plt.imshow(img)

# fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True)
# # print(fd.shape)
# for i in range(len(hog_image)):
#     for j in range(len(hog_image[0])):
#         if hog_image[i][j] != 0:
#             print(hog_image[i][j])

# # print(hog_image)

# plt.axis("off")
# plt.imshow(hog_image, cmap="gray")
# plt.show()
# cv2.imwrite("hog.png", hog_image*255)