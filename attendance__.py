import pathlib
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = "images"
images = []
person_name = []
myList = os.listdir(path)

# print(myList)

def people_name():
    for curr_image in myList:
        current_image = cv2.imread(f'{path}/{curr_image}')
        images.append(current_image)
        file_extension = pathlib.Path(curr_image).suffix
        names = curr_image.removesuffix(file_extension)
        person_name.append(names)
    # print(person_name)
people_name()

def faceEncoding(images):
    global encode_list
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list
def markAttendence(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readline()
        nameList = []
        for line in myDataList:
            entry = line.split((','))
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now .strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
            return
encode_data = faceEncoding(images)
print("face data fetch succesfully")

cap = cv2.VideoCapture(0)

def face_rec():
    while True:
        ret,frame = cap.read()
        faces = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame_faces = face_recognition.face_locations(faces)
        faces_encode = face_recognition.face_encodings(faces,frame_faces)
        for encodeFace,faceLoc in zip(faces_encode,frame_faces):
            matches = face_recognition.compare_faces(encode_data,encodeFace)
            face_dis = face_recognition.face_distance(encode_data,encodeFace)
            matchIndex = np.argmin(face_dis)
            if matches[matchIndex]:
                nme = person_name[matchIndex].upper()
                # print(nme)
                y1,x2,y2,x1 = faceLoc
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,str(nme), (x1+13,y2+13), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0,255), 2,cv2.LINE_AA, False)
                markAttendence(nme)
        cv2.imshow("camera",frame)
        if (cv2.waitKey(1) == ord("q")):
            encode_list.clear()
            break
    cap.release()
    cv2.destroyAllWindows()
face_rec()