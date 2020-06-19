import cv2
import numpy as np
import os

threshold = 10
orb = cv2.ORB_create()
path = "images/train_images"
images = []
classNames = []
myList = os.listdir(path)


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

    if threshold < max(matchList):
        max_index = matchList.index(max(matchList))

    print("Match List: ", matchList, classNames[np.argmax(matchList)], max_index, end=" ")
    return max_index


desList = findDes(images)

cap = cv2.VideoCapture(0)

while True:

    success, img = cap.read()
    imgOriginal = img.copy()
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    id = findID(gray_image, desList)

    if id == -1:
        cv2.putText(
            gray_image,
            "NOT FOUND!",
            (50, 50), cv2.FONT_HERSHEY_COMPLEX,
            1, (0, 255, 0), 2
        )
    else:
        cv2.putText(
            gray_image,
            classNames[id],
            (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0), 2
        )

    cv2.imshow("Result", gray_image)

    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()











