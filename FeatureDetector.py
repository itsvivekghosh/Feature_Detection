import cv2
import numpy as np

img1 = cv2.imread("images/train images/Redmi/Redmi-Note-5-Pro-Product-shots-1.jpg", 0)
img2 = cv2.imread("images/train images/Redmi/Xiaomi-Redmi-Note-5-Pro-review-tech2-master-1280.jpg", 0)

img1 = cv2.resize(img1, (250, 500))
img2 = cv2.resize(img2, (250, 500))
orb = cv2.ORB_create()

kp1, dec1 = orb.detectAndCompute(img1, None)
kp2, dec2 = orb.detectAndCompute(img2, None)

# imgKp1 = cv2.drawKeypoints(img1, kp1, None)
# imgKp2 = cv2.drawKeypoints(img2, kp2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(dec1, dec2, k=2)

# print(matches)
# print(dec1.shape)
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

result = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

cv2.imshow("Image", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
