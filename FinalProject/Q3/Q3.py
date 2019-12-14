# Calibrate the Camera using the OpenCV calibrateCamera tool
import numpy as np
import cv2 as cv
import glob
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((5*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:5].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('*.jpg')
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,5), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,5), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(50)
cv.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

axis = np.float32([[0,0,0], [0,2,0], [2,2,0], [2,0,0],[0,0,-2],[0,2,-2],[2,2,-2],[2,0,-2]])

# Draw the wire frame with corresponding points:
def draw(img, corners, imgpts, attach=False):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw pillars in blue color
    for i,j in zip([0,1,2,3],[4,5,6,7]):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    for i,j in zip([0,2,4,6],[1,3,5,7]):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    for i,j in zip([0,1,4,5],[3,2,7,6]):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    return img


for fname in glob.glob('*.jpg'):
    img = cv.imread(fname)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (7,5), None)
    if ret == True:
        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # Find the rotation and translation vectors.
        ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
        # project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
        img = draw(img, corners2, imgpts)

        selffie_img = cv.imread("C:\\Users\\wangz\\Documents\\GitHub\\ECE561MachineVision\\FinalProject\\selffie.jpg")
        selffie_img = cv.resize(selffie_img, (1000,1000))
        selffie_img = cv.cvtColor(selffie_img , cv.COLOR_RGB2BGR)
        back_img = img
        rows , cols ,ch = selffie_img.shape
        back_rows , back_cols ,back_ch = back_img.shape
        # persective transform according to the projected imgpts
        # I selected the top facet as imgpts[4:]
        pts1 = np.float32([[0,0],[rows,0],[rows,cols],[0,cols]])
        pts2 = np.float32([imgpts[4:][3], imgpts[4:][0], imgpts[4:][1], imgpts[4:][2]])
        M = cv.getPerspectiveTransform(pts1,pts2)
        dst = cv.warpPerspective(selffie_img,M,(back_cols,back_rows))
        # convert my selffie to gray in order to fit mask
        # use a binary mask to achieve a bitwise-add
        gray_dst = cv.cvtColor(dst , cv.COLOR_RGB2GRAY)
        ret,mask = cv.threshold(gray_dst,1,255,cv.THRESH_BINARY)
        mask_inv = cv.bitwise_not(mask)
        # attach the selffie using bitwise_and
        img_bg = cv.bitwise_and(back_img, back_img, mask = mask_inv)
        res = cv.add(img_bg, dst)

        cv.imshow('img',res)
        k = cv.waitKey(0) & 0xFF
        if k == ord('s'):
            cv.imwrite(fname[:-4]+'.png', img)
cv.destroyAllWindows()

