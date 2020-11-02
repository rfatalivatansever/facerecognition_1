import cv2

eye   = cv2.CascadeClassifier("haarcascade_eye.xml")
#smile = cv2.CascadeClassifier("haarcascade_smile.xml")
face  = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#body  = cv2.CascadeClassifier("haarcascade_fullbody.xml")

img = cv2.imread("image.jpg")

graycolor = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = face.detectMultiScale(graycolor,1.3,3)
for (x,y,w,h) in faces :
    img   = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    gray  = graycolor[y:y+h, x:x+w]
    color = img[y:y+h, x:x+w]

eyes = eye.detectMultiScale(graycolor,1.3,1)
for (x,y,w,h) in eyes :
    img   = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
    gray  = graycolor[y:y + h, x:x + w]
    color = img[y:y + h, x:x + w]
"""
smiles = smile.detectMultiScale(graycolor,1.1,1)
for (x,y,w,h) in smiles :
    img   = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
    gray  = graycolor[y:y + h, x:x + w]
    color = img[y:y + h, x:x + w]

body_1 = body.detectMultiScale(graycolor,1.1,1)
for (x,y,w,h) in smiles :
    img   = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
    gray  = graycolor[y:y + h, x:x + w]
    color = img[y:y + h, x:x + w]
"""


cv2.imshow("Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()