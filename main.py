import pickle
import numpy as np
import cv2
import os
import cvzone
import face_recognition
import serial
import time
from EencodeGenerator import encodeListKnownWithIds


#arduino = serial.Serial('COM5', 9600)  # Replace COM3 with your Arduino port
#time.sleep(2)  # Wait for connection to initialize





cap=cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4,480 )
imgBackground=cv2.imread('Resources/background.png')

#importing the mode images into a list


folderModPath='Resources/Modes'
modePathList=os.listdir(folderModPath)
imgModeList=[]
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModPath,path)))


#import the encoding file
print("loading the encode file . . . .")
file=open('EncodeFile.p','rb')
encodeListKnownWithIds=pickle.load(file)
file.close()
encodeListKnown,studentIds=encodeListKnownWithIds
print("encode file loaded ")
#print(studentIds)


while True :
    success,img=cap.read()
    imgs =cv2.resize(img ,(0,0),None,0.25,0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
    faceCurFrame=face_recognition.face_locations(imgs)
    encodeCurFame=face_recognition.face_encodings(imgs,faceCurFrame)




    imgBackground[162:162+480,55:55+640 ] =img
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[2]
    for encodeFace,faceLoc in zip(encodeCurFame,faceCurFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)
        #print("matches",matches)
       # print("faceDis",faceDis)
        matchIndex=np.argmin(faceDis)
        #print("match Index",matchIndex)
        if matches[matchIndex]:
            print("face known detected")
            print(studentIds[matchIndex])
            #arduino.write(b'UNLOCK\n')
    #cv2.imshow("webcam",img)

    cv2.imshow("face attendence",imgBackground)

    cv2.waitKey(1)