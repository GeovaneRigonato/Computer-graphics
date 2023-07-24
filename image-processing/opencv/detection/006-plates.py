import cv2
import numpy as np
import pytesseract


plateCascade = cv2.CascadeClassifier('haarcascades/haarcascade_russian_plate_number.xml')

#5, #7 -> Não funciona
#9 -> Argentina
img = cv2.imread('images/6.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plates = plateCascade.detectMultiScale(gray)
print(plates)

i = 0
for (x,y,w,h) in plates:
    cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
    plate = gray[y: y+h, x:x+w]
    i+=1
    cv2.imshow("plate"+ str(i), plate)

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
_, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

config_tesseract = "--tessdata-dir tessdata --psm 6"
plate_number = pytesseract.image_to_string(plate, lang="por", config="--psm 6")

if plate_number:
    print("Número da placa:", plate_number)
else:
    print("Não foi possível reconhecer o número da placa.")