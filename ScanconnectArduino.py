import os.path
import cv2
import serial
import pyttsx3
import time

camera = 0
video = cv2.VideoCapture(camera, cv2.CAP_DSHOW)

file_exists = os.path.exists('DataSet/Training.xml')
print(file_exists)


if file_exists is False:
    exec(open("Training.py").read())
else:
    pass


engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
engine.setProperty('rate', 150)
engine.runAndWait()

def speak(str):
    engine.say(str)
    engine.runAndWait()

#ser = serial.Serial("COM5", 9600, timeout = 1)

#def retrieveData():
    #ser.write(b'1')
    #data = ser.readline()
    #return data

speak("Welcome to safety driving system. Camera is opening for 3 seconds")
time.sleep(2)

while True:
    faceDeteksi = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('DataSet/Training.xml')
    a = 0
    a = a+1
    check, frame = video.read()
    abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    wajah = faceDeteksi.detectMultiScale(abu,1.3,5)
    for (x,y,w,h) in wajah:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),4)
        id, conf = recognizer.predict(abu[y:y + h, x:x + w])
        cv2.putText(frame,str(id),(x+20,y-20),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0))

        #(retrieveData())

    #else:
        #ser.write(b'0')


    cv2.imshow("Face Recognition",frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

