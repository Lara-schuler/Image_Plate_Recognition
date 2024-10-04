import cv2
import os
from vehicle_detection import detect_vehicle
from plate_detection import detect_plate
from ocr_recognition import recognize_plate

# Caminhos para os arquivos do YOLO
yolo_cfg = r'C:\yolo\yolov4-tiny.cfg'
yolo_weights = r'C:\yolo\yolov4-tiny.weights'

# Caminho da imagem de entrada
image_path = r'C:/images/carro4.jpg'

# Carregar a imagem
image = cv2.imread(image_path)

# Verifica se a imagem foi carregada corretamente
if image is None:
    print(f"Erro ao carregar a imagem: {image_path}")
    exit()

# Detectar veículo na imagem
vehicle_img, vehicle_coords = detect_vehicle(image, yolo_cfg, yolo_weights)

# Se um veículo foi detectado
if vehicle_img is not None:
    print("Veículo detectado.")

    # Detectar a placa no veículo
    plate_img, plate_coords = detect_plate(vehicle_img, yolo_cfg, yolo_weights)

    # Se uma placa foi detectada
    if plate_img is not None:
        print("Placa detectada.")
        
        # Reconhecimento OCR na placa
        plate_text = recognize_plate(plate_img)
        
        if plate_text:
            print(f"Texto da placa: {plate_text}")
        else:
            print("Erro ao reconhecer a placa.")
    else:
        print("Nenhuma placa detectada.")
else:
    print("Nenhum veículo detectado.")
