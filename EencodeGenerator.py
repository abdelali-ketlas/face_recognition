import cv2
import face_recognition
import pickle
import os


#importing students images
folderPath='images'
PathList=os.listdir(folderPath)
print(PathList)
imgList=[]
studentIds=[]
for path in PathList:
    imgList.append(cv2.imread(os.path.join(folderPath,path)))
    studentIds.append(os.path.splitext(path)[0])
   # print(os.path.splitext(path)[0])
print(studentIds)

def findEncodings(imagesList):
    encodeList=[]
    for img in imagesList :
      img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
      encode=face_recognition.face_encodings(img)[0]
      encodeList.append(encode)
    return encodeList
print("encoding started . . . . ")
encodeListKnown=findEncodings(imgList)
encodeListKnownWithIds=[encodeListKnown,studentIds]
print("encoding complete")

file=open("EncodeFile.p",'wb')
pickle.dump(encodeListKnownWithIds,file)
file.close()
print("file saved")
