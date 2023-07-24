import cv2
import imutils
import numpy as np

def adjust_contrast(image, alpha, beta):
    # Ajuste de contraste linear: new_pixel_value = alpha * pixel_value + beta
    adjusted_image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)
    return adjusted_image

# Carregar a imagem
image = cv2.imread("images/carro2.png")
cv2.imshow("1 original", image)

# Aumentar o contraste da imagem
alpha = 1.5  # Fator de aumento do contraste (1.0 significa nenhum ajuste)
beta = 50    # Valor adicionado a cada pixel após o ajuste de contraste
image_contrast = adjust_contrast(image, alpha, beta)
cv2.imshow("3 contraste aumentado", image_contrast)

# Converter para escala de cinza
image_gray = cv2.cvtColor(image_contrast, cv2.COLOR_BGR2GRAY)
cv2.imshow("2 gray", image_gray)


# Aplicar desfoque gaussiano para suavizar os contornos
blurred_edges = cv2.GaussianBlur(image_contrast, (9, 9), 0)

# Binarizar a imagem
_, image_thresh = cv2.threshold(blurred_edges, 128, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("4 limiar", image_thresh)

# Encontrar contornos na imagem binarizada
contours, _ = cv2.findContours(image_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtrar os contornos por área para obter apenas a placa
min_area = 1000  # Ajuste este valor conforme necessário
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

# Encontrar o maior contorno, que deve ser a placa
largest_contour = max(filtered_contours, key=cv2.contourArea)

# Criar uma imagem em branco para desenhar a placa
plate_image = np.zeros_like(image_gray)

# Desenhar o contorno da placa na imagem em branco
cv2.drawContours(plate_image, [largest_contour], -1, 255, thickness=cv2.FILLED)

# Aplicar máscara para extrair a placa da imagem original
plate = cv2.bitwise_and(image, image, mask=plate_image)

cv2.imshow("Placa extraída", plate)

cv2.waitKey(0)
cv2.destroyAllWindows()
