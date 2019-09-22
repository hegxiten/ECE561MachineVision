import numpy as np
import cv2
from matplotlib import pyplot as plt

# change the directory below for any input/output
SRC_FILE = 'hw2_q8_side.jpg'
DST_FILE = 'hw2_q8_front.jpg'

img_side = cv2.imread(SRC_FILE, cv2.IMREAD_GRAYSCALE)          # SideView
img_front = cv2.imread(DST_FILE, cv2.IMREAD_GRAYSCALE)         # FrontView

# selecting four matching points from each view
# stored it in src, dst respectively
plt.imshow(img_side)
src = side_keypoints = np.array(plt.ginput(4))
plt.close()
plt.imshow(img_front)
dst = front_keypoints = np.array(plt.ginput(4))
plt.close()

A = []
for i in range(0, len(src)):
    x, y = src[i][0], src[i][1]
    u, v = dst[i][0], dst[i][1]
    A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
    A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
A = np.asarray(A)
U, S, Vh = np.linalg.svd(A)
L = Vh[-1,:] / Vh[-1,-1]
H = L.reshape(3, 3)
print("Homography Matrix:")
print(H)
# warpping the source image to the output
im_out = cv2.warpPerspective(img_side, h, (img_front.shape[1],img_front.shape[0]))
# display images with reduced size 720*540
im_to_show = cv2.resize(im_out, (720,540))
cv2.imshow("warpped", im_to_show)
cv2.waitKey(0)
cv2.destroyAllWindows()