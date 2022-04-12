import cv2
import os

camera = 0
video = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
faceDeteksi = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
id = input('Masukan id : ')
nama = input('Masukkan Nama : ')

if os.path.exists('DataSet/Training.xml'):
    os.remove('DataSet/Training.xml')
    print('The file has been deleted succesfully')
else:
    print('The file doesnt exist')

a = 0

while True:
    a = a+1
    check, frame = video.read()
    abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    wajah = faceDeteksi.detectMultiScale(abu,1.3,5)
    for (x,y,w,h) in wajah:
        cv2.imwrite('DataSet/User.'+str(id)+'.'+str(a)+'.'+str(nama)+'.jpg',abu[y:y+h,x:x+w])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        roi_warna = frame[y:y + h, x:x + w]
        roi_abu = abu[y:y + h, x:x + w]
    cv2.imshow ("Face Recognition",frame)
    if (a>19):
        break

video.release()
cv2.destroyAllWindows()
