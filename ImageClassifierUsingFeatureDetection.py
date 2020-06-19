import cv2
import numpy as np
import os

threshold = 20
orb = cv2.ORB_create()
path = "images/train_images"
images = []
classNames = []
myList = os.listdir(path)


# print(myList)
print("Total Classes Detected: {}".format(len(myList)))

for class_ in myList:
    imgCur = cv2.imread(f"{path}/{class_}", 0)
    images.append(imgCur)
    classNames.append(os.path.splitext(class_)[0])
print(classNames)


def findDes(images):

    desList = []

    for image in images:
        kp, des = orb.detectAndCompute(image, None)
        desList.append(des)

    return desList


def findID(image, desList):

    kp2, des2 = orb.detectAndCompute(image, None)
    bf = cv2.BFMatcher()
    max_index = -1
    matchList = []

    for des in desList:
        matches = bf.knnMatch(des, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])

        matchList.append(len(good))
    print("Match List: ", matchList, np.argmax(matchList), classNames[np.argmax(matchList)])

    if threshold < max(matchList):
        max_index = matchList.index(max(matchList))

    return max_index

desList = findDes(images)

cap = cv2.VideoCapture(0)

while True:

    success, img = cap.read()
    imgOriginal = img.copy()
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.putText(gray_image, classNames[findID(gray_image, desList)], (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 0), 2)

    cv2.imshow("Result", gray_image)

    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()











