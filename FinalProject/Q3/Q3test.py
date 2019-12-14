import cv2
import numpy as np
import matplotlib.pyplot as plt


photo_img = cv2.imread("photo.png")
back_img = cv2.imread("advert.png")

rows , cols ,ch = photo_img.shape
back_rows , back_cols ,back_ch = back_img.shape


# opencv 通道转换为 plt通道格式
photo_img = cv2.cvtColor(photo_img , cv2.COLOR_RGB2BGR)
back_img = cv2.cvtColor(back_img , cv2.COLOR_RGB2BGR)


# 透视转换计算
pts1 = np.float32([[0,0],[rows,0],[rows,cols],[0,cols]])
pts2 = np.float32([[462,126],[813,216],[812,643],[442,604]])

M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(photo_img,M,(back_cols,back_rows))



# 制作掩模 Mak
gray_dst = cv2.cvtColor(dst , cv2.COLOR_RGB2GRAY)
ret,mask =cv2.threshold(gray_dst,1,255,cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

img_bg = cv2.bitwise_and(back_img,back_img,mask = mask_inv)

res = cv2.add(img_bg,dst)


# plt显示图像
plt.subplot(131),plt.imshow(photo_img),plt.title('Photo')
plt.subplot(132),plt.imshow(back_img),plt.title('Advert')
plt.subplot(133),plt.imshow(res),plt.title('Result')
plt.show()


# 保存合成照片
save_res = cv2.cvtColor(res , cv2.COLOR_BGR2RGB)
cv2.imwrite('res.png',save_res)

cv2.waitKey(10000)
cv2.destroyAllWindows()