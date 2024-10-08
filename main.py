import cv2
import os
from vehicle_detection import detect_vehicle
from plate_detection import detect_and_recognize_plate  # Atualizando para usar a nova função com OpenALPR

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

    # Detectar e reconhecer a placa no veículo usando OpenALPR e pré-processamento
    plate_text = detect_and_recognize_plate(vehicle_img)

    # Se uma placa foi reconhecida
    if plate_text:
        print(f"Placa reconhecida: {plate_text}")
    else:
        print("Nenhuma placa reconhecida.")
else:
    print("Nenhum veículo detectado.")
