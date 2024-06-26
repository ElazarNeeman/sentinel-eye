import os
from operator import itemgetter

import cv2
from deepface import DeepFace

from identity import get_name

img1 = r"test_images\test-faces.jpg"
img = cv2.imread(img1)
faces = DeepFace.extract_faces(img, align=True)

for f in faces:
    x, y, w, h, left_eye, right_eye = itemgetter('x', 'y', 'w', 'h', 'left_eye', 'right_eye')(f['facial_area'])
    # print(x, y, w, h)
    img = cv2.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
    res = DeepFace.find(img_path=f['face'], db_path="im_db_judges", align=False, enforce_detection=False, silent=True)

    name = get_name(res)
    if name is None:
        name = "?"

    cv2.putText(img, name.upper(), (x, y - 10),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255, 0, 0), 2)

    print(name)

cv2.imshow("Processed", img)
cv2.waitKey(0)
name = input("Press enter to exit")
