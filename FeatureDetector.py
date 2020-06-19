import cv2
import numpy as np

img1 = cv2.imread("images/train images/Redmi-Note-5-Pro-Product-shots-1.jpg", 0)
img2 = cv2.imread("images/train images/powerbank.jpg", 0)
img3 = cv2.imread("images/train images/touch-screen-and-lcd-for-xiaomi-redmi-5.jpg", 0)

img1 = cv2.resize(img1, (250, 500))
img2 = cv2.resize(img2, (250, 500))
img3 = cv2.resize(img3, (250, 500))
orb = cv2.ORB_create()

kp1, dec1 = orb.detectAndCompute(img1, None)
kp2, dec2 = orb.detectAndCompute(img2, None)
kp3, dec3 = orb.detectAndCompute(img3, None)

imgKp1 = cv2.drawKeypoints(img1, kp1, None)
imgKp2 = cv2.drawKeypoints(img2, kp2, None)
imgKp3 = cv2.drawKeypoints(img3, kp3, None)

print(dec1.shape)
cv2.imshow("Image", imgKp2)
cv2.waitKey(0)
cv2.destroyAllWindows()
