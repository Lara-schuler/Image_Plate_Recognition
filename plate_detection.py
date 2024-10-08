import cv2
import numpy as np
import openalpr  # Certifique-se de que essa é a importação correta para a sua instalação
import os

# Verifique se as pastas existem, se não, crie-as
os.makedirs('plates', exist_ok=True)
os.makedirs('output', exist_ok=True)

def preprocess_plate(plate_image):
    # Converte a imagem para escala de cinza
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    
    # Binarização usando Otsu
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Filtragem para remover ruído
    filtered = cv2.medianBlur(binary, 3)

    # Equalização de histograma
    equalized = cv2.equalizeHist(filtered)

    return equalized

def recognize_plate_openalpr(plate_image):
    # Caminho para as DLLs do OpenALPR
    openalpr_path = r'C:\openalpr_64'
    os.environ['PATH'] += os.pathsep + openalpr_path  # Adiciona o caminho ao PATH
    

    # Use o caminho completo para a DLL
    dll_path = os.path.join(openalpr_path, "libopenalpr.dll")
    
    # Instanciar o OpenALPR com o caminho da DLL
    alpr = openalpr.Alpr("br", r"C:\openalpr_64\openalpr-master\runtime_data\config\openalpr.conf", r"C:\openalpr_64\openalpr-master\runtime_data")

    if not alpr.is_loaded():
        print("Erro ao carregar o OpenALPR. Verifique se a DLL está no caminho correto.")
        return None

    alpr.set_top_n(1)
    alpr.set_default_region("br")

    # Processar a imagem da placa
    results = alpr.recognize_ndarray(plate_image)

    # Se encontrar resultados, retornar o número da placa
    if len(results['results']) > 0:
        plate_text = results['results'][0]['plate']
        print(f"Placa reconhecida: {plate_text}")
        return plate_text
    else:
        print("Nenhuma placa reconhecida com OpenALPR.")
        return None

def detect_and_recognize_plate(vehicle_img):
    # Aplicar técnicas de segmentação e encontrar a área da placa
    gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)

        if 2 < aspect_ratio < 5 and 100 < w < 500 and 40 < h < 200:
            plate_img = vehicle_img[y:y + h, x:x + w]

            # Pré-processar a imagem da placa
            processed_plate = preprocess_plate(plate_img)

            # Reconhecer os caracteres da placa com OpenALPR
            plate_text = recognize_plate_openalpr(processed_plate)

            # Salvar a imagem da placa segmentada
            cv2.imwrite(f'plates/placa_detectada_{len(os.listdir("plates")) + 1}.jpg', plate_img)
            print(f"Placa segmentada salva em: plates/placa_detectada_{len(os.listdir('plates'))}.jpg")

            return plate_text
    print("Nenhuma placa detectada.")
    return None
