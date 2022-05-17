from base64 import encode
from importlib.resources import path
from tarfile import ENCODING
import cv2
from cv2 import findTransformECC
import face_recognition
from matplotlib.pyplot import show
import numpy as np
import os
from datetime import datetime

#path is equal to the name of directory
path = 'images'

#list of images stored
images=[]

#list of name stored
personName=[]

#listing of current directory components
myList=os.listdir(path)
myList.remove('.DS_Store')

#returning images in image directory
print (myList)

for cur_img in myList:

    #reading images
    current_img=cv2.imread(f'{path}/{cur_img}')

    #adding images in the list
    images.append(current_img)

    personName.append(os.path.splitext(cur_img)[0])
print (personName)


def faceEncodings(images):

    encodeList=[]

    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    #HOG algorithm
    return (encodeList)

encodListKnown=faceEncodings(images)

print("All encodings completed !!!!")


#Marking Attendance
def attendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')#recording time
            dStr = time_now.strftime('%d/%m/%Y')#recording date
            f.writelines(f'\n{name},{tStr},{dStr}')
#reading camera
cap=cv2.VideoCapture(0)

while True:
    ret, frame=cap.read()

    #resizing frame due to resolution differences
    faces=cv2.resize(frame,(0,0),None,0.25,0.25)

    #BGR to RGB as cv2 returns BGR format
    faces=cv2.cvtColor(faces,cv2.COLOR_BGR2RGB)
    
    #determining location of faces in current frame
    facesCurrentFrame = face_recognition.face_locations(faces)

    #encoding current frame
    encodeCurrentFrame = face_recognition.face_encodings(faces,facesCurrentFrame)
    
    #matching current frame and the encoded frame
    for encodeFace, faceLoc in zip(encodeCurrentFrame,facesCurrentFrame):
        matches = face_recognition.compare_faces(encodListKnown,encodeFace)
        facedis = face_recognition.face_distance(encodListKnown,encodeFace)
        
        #finding the index value of minimum distance
        matchIndex = np.argmin(facedis)

        if matches[matchIndex]:
            name=personName[matchIndex].upper()
            
            y1,x2,y2,x1=faceLoc

            #Considering resizing parameters
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4

            #Drawing rectangle
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

            #Rectangle with name
            cv2.rectangle(frame,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)

            #putting text inside the frame
            cv2.putText(frame,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            attendance(name)




    
    cv2.imshow("Camera",frame)

    #ASCII code of enter key=13
    if cv2.waitKey(10) == 13:
        break

cap.release()
cv2.destroyAllWindows()