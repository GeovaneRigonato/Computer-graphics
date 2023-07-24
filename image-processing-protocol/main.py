import cv2 as cv
import pytesseract
import os

    # cv.imshow('imagem',img)

    # new_height = 450
    # new_width = 600

    # img_redimensionada = cv.resize(img, (new_width, new_height))
    # cv.imshow('imagem redimensionada',img_redimensionada)

def encontrarRoiPlaca(source, count):
    img = cv.imread(source)
    
    if source == 'imagens/policia.png':
      gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
      cv.imshow('gray',gray)

      _, binary = cv.threshold(gray, 90, 255, cv.THRESH_BINARY)
      cv.imshow('binary',binary)
      
      contornos, hierarquia = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
      for c in contornos:
        perimetro = cv.arcLength(c, True) # apenas os fechados
      
        if(perimetro > 120):
        
          aprox = cv.approxPolyDP(c, 0.03 * perimetro, True)
          if len(aprox) == 4:
            (x,y,w,h) = cv.boundingRect(aprox)
            rectangles = cv.rectangle( img, (x,y), (x+w,y+h), (0,255,0), 2)
            plate = img[y:y+h, x:x+w]
            
            cv.imwrite(f'resultado/plate_{count}.jpg', plate)
            count += 1
          
          
          cv.imshow('contornos',img)
          return
    else:
  
      gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
      cv.imshow('gray',gray)

      _, binary = cv.threshold(gray, 90, 255, cv.THRESH_BINARY)
      cv.imshow('binary',binary)

      desfoque = cv.GaussianBlur(binary, (5, 5), 0)
      cv.imshow('desfoque',desfoque)

      contornos, hierarquia = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

      # cv.drawContours(img, contornos, -1, (0, 255, 0), 2)
      # cv.imshow('contornos',img)

      for c in contornos:
        perimetro = cv.arcLength(c, True) # apenas os fechados
        
        if(perimetro > 120):
        
          aprox = cv.approxPolyDP(c, 0.03 * perimetro, True)
          if len(aprox) == 4:
            (x,y,w,h) = cv.boundingRect(aprox)
            rectangles = cv.rectangle( img, (x,y), (x+w,y+h), (0,255,0), 2)
            plate = img[y:y+h, x:x+w]
            
            cv.imwrite(f'resultado/plate_{count}.jpg', plate)
            count += 1
            
      cv.imwrite('contorno/contorno.jpg', img)
      cv.imshow('contornos',img)
    


def preProcessamento(max):
    
    for i in range(max):
      img_roi = cv.imread(f'resultado/plate_{i}.jpg')
      if img_roi is None:
        return  
      cv.imshow(f'roi_{i}',img_roi)
      
      resize = cv.resize(img_roi, None, fx=8, fy=8, interpolation=cv.INTER_CUBIC)
      
      gray = cv.cvtColor(resize, cv.COLOR_BGR2GRAY)
      _, binary = cv.threshold(gray, 90, 255, cv.THRESH_BINARY)
      
      cv.imwrite(f'resultado/roi-ocr_{i}.jpg', binary)
      
      cv.imshow('res',binary)
      


def ocrImageRoiPlate(count):
  for i in range(count):
    img_roi = cv.imread(f'resultado/roi-ocr_{i}.jpg')
    if img_roi is None:
        return  
    
    config = r'-c tessedit_char_whitelist=-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6'
    
    saida = pytesseract.image_to_string(img_roi, lang='eng', config=config)
    
    if saida.strip() != '':
      print (f'Placa : {saida.strip()}')

# imgPlate = cv.imread('resultado/plate.jpg')

# gray = cv.cvtColor(imgPlate, cv.COLOR_BGR2GRAY)

# _, binary = cv.threshold(gray, 90, 255, cv.THRESH_BINARY)

# desfoque = cv.GaussianBlur(binary, (5, 5), 0)
# cv.imshow('desfoque',desfoque)

if __name__ == "__main__":
    count = 1
    source = 'images/funciona.jpg'
    max = 20
    encontrarRoiPlaca(source, count-1)
    cv.destroyAllWindows()
    preProcessamento(max)
    cv.destroyAllWindows()
    ocrImageRoiPlate(count)
    cv.destroyAllWindows()
    
    print('.........Digite qualquer tecla para sair..........')
    cv.waitKey(0)
    
    folder = 'resultado'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Erro ao excluir o arquivo {file_path}: {e}")
    
    os.remove('contorno/contorno.jpg')