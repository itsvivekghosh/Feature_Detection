import cv2
import numpy as np

# img1 = cv2.imread("images/train_images/Redmi 5-1.jpg", 0)
# img2 = cv2.imread("images/test_images/Redmi-Note-5-Pro-Product-shots-1.jpg", 0)

img1 = cv2.imread("images/train_images/Redmi 5.jpg", 0)
img2 = cv2.imread("images/test_images/Redmi-Note-5-Pro-Product-shots-1.jpg", 0)

orb = cv2.ORB_create(nfeatures=1000)

kp1, dec1 = orb.detectAndCompute(img1, None)
kp2, dec2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(dec1, dec2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

result = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
print(len(good))

cv2.imshow("Image", img1)
cv2.imshow("Image", img2)
cv2.imshow("Image", result)
cv2.imwrite("result2.jpg", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
