import cv2

img = cv2.imread('big_files/HFOUG.jpg',cv2.IMREAD_GRAYSCALE)
_,img = cv2.threshold(img,0,255,cv2.THRESH_OTSU)
h, w = img.shape[:2]

contours0, hierarchy, _ = cv2.findContours( img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
moments  = [cv2.moments(cnt) for cnt in contours0]
# Nota Bene: I rounded the centroids to integer.
centroids = []
for m in moments:
    if m['m00'] != 0:
        centroids.append((int(round(m['m10']/m['m00'])),int(round(m['m01']/m['m00']))))

# centroids = [( int(round(m['m10']/m['m00'])),int(round(m['m01']/m['m00'])) ) for m in moments]

print('cv2 version:', cv2.__version__)
print('centroids:', centroids)

for c in centroids:
    # I draw a black little empty circle in the centroid position
    cv2.circle(img,c,5,(0,0,0))

cv2.imshow('image', img)
0xFF & cv2.waitKey(27)
cv2.destroyAllWindows()
