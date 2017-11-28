import cv2
import numpy as np

def transform(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    myFrom = np.float32([
            [1042, 406],
            [1299, 406],
            [903, 791],
            [356, 790],
        ])

    sL = 40
    tW = 2000
    tH = 2000

    myTo = np.float32([
            [tW/2 - 3*sL, tH/2 - 20*sL],
            [tW/2 + 3*sL, tH/2 - 20*sL],
            [tW/2 + 3*sL, tH/2 + 20*sL],
            [tW/2 - 3*sL, tH/2 + 20*sL],
        ])
    tMat = cv2.getPerspectiveTransform(myFrom,myTo)

    dst = cv2.warpPerspective(img, tMat, (tW, tH))

    return (tMat, dst)
