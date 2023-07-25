import cv2
import numpy as np
import pytesseract

def increase_saturation(image, saturation_factor):
    # Converter a imagem de BGR para HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Aumentar a saturação multiplicando o canal S pelo fator de saturação
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 255)

    # Converter a imagem de volta para BGR
    result_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return result_image

plateCascade = cv2.CascadeClassifier('haarcascades/haarcascade_russian_plate_number.xml')

img = cv2.imread('images/Car_teste19.jpg')
#img_saturated = increase_saturation(img, 1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plates = plateCascade.detectMultiScale(gray)

i = 0
for (x,y,w,h) in plates:
    cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
    plate = gray[y: y+h, x:x+w]
    i+=1
    cv2.imshow("plate"+ str(i), plate)

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()

blurred_image = cv2.GaussianBlur(plate, (5, 5), 0)
resized_image = cv2.resize(plate, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
_, binary_img = cv2.threshold(plate, 90, 240, cv2.THRESH_BINARY)

config_tesseract = '-c tessedit_char_whitelist=-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6'
plate_number = pytesseract.image_to_string(binary_img, lang="por", config=config_tesseract)

if plate_number:
    print("Número da placa:", plate_number)
    cv2.imwrite('images/result/result.jpeg', binary_img)
else:
    print("Não foi possível reconhecer o número da placa.")