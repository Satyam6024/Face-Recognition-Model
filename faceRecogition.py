import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = "AttendenceImages"
images = []
className = []
myList = os.listdir(path)
print(myList)

for cls in myList:
    currentImg = cv2.imread(f'{path}/{cls}')
    images.append(currentImg)
    className.append(os.path.splitext(cls)[0])
print(className)


def findEncodings(images):
    encodedList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodedList.append(encode)
    return encodedList

def markAttendence(name):
    with open('Attendence.csv','r+') as f:
        myDataList = f.readline()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


encodeKnownList = findEncodings(images)
print('Encoding Completed...')

capture = cv2.VideoCapture(0)

while True:
    successs, img = capture.read()
    imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    faceCurrentFrame = face_recognition.face_locations(imgSmall)
    encodeCurrentFrame = face_recognition.face_encodings(imgSmall, faceCurrentFrame)

    for encodeFace, faceLocation in zip(encodeCurrentFrame, faceCurrentFrame):
        matches = face_recognition.compare_faces(encodeKnownList, encodeFace)
        faceDistance = face_recognition.face_distance(encodeKnownList, encodeFace)
        print(faceDistance)
        matchIndex = np.argmin(faceDistance)

        if matches[matchIndex]:
            name = className[matchIndex].upper()
            #print(name)

            y1, x2, y2, x1 = faceLocation
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
            markAttendence(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
