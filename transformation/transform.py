import cv2
import numpy as np

def transform(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # pixel coordinates of calibration checkerbox corners in original picture
    myFrom = np.float32([
            [1042, 406],
            [1299, 406],
            [903, 791],
            [356, 790],
        ])

    # side length of a square
    sL = 40
    # width and height of output transformed image
    tW = 2000
    tH = 2000

    # pixel coordinates of the calibration checkerbox corners in output transformed image
    myTo = np.float32([
            [tW/2 - 3*sL, tH/2 - 20*sL],
            [tW/2 + 3*sL, tH/2 - 20*sL],
            [tW/2 + 3*sL, tH/2 + 20*sL],
            [tW/2 - 3*sL, tH/2 + 20*sL],
        ])

    # Get transformation matrix
    tMat = cv2.getPerspectiveTransform(myFrom,myTo)

    # warp input image with transformation matrix (use this to warp detected centroids)
    dst = cv2.warpPerspective(img, tMat, (tW, tW))

    return (tMat, dst)

def main():
    img = cv2.imread('betterCheckb.png')
    transformationMatrix, sampleTransform = transform(img)
    cv2.imshow("transformed", sampleTransform)
    cv2.waitKey()

if __name__ == '__main__':
    main()