import os
import cv2
import numpy as np
from yolo_utils import load_yolo

# Verifique se as pastas existem, se não, crie-as
os.makedirs('plates', exist_ok=True)
os.makedirs('output', exist_ok=True)

def visualize_images(original, plate_img, processed_plate):
    # Cria uma imagem combinada para visualização
    combined_image = np.hstack((original, plate_img, processed_plate))
    cv2.imshow("Original | Detecção da Placa | Placa Processada", combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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

def detect_plate(image, model_cfg, model_weights, confidence_threshold=0.5, nms_threshold=0.4):
    # Carregar o modelo YOLO
    net, output_layers = load_yolo(model_cfg, model_weights)

    if net is None or output_layers is None:
        print("Falha ao carregar o modelo YOLO.")
        return None, None

    height, width, _ = image.shape

    # Pré-processamento da imagem para o YOLOv4-tiny
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # Processamento da saída
    boxes = []
    confidences = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    # Aplica Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    if len(indices) > 0:
        for i in indices:
            i = i[0]
            box = boxes[i]
            x, y, w, h = box
            
            # Recortar a imagem da placa
            plate_img = image[y:y + h, x:x + w]

            # Aplicar o pré-processamento
            processed_plate = preprocess_plate(plate_img)

            # Salvar a imagem da placa original na pasta 'plates'
            cv2.imwrite(f'plates/placa_original_{len(os.listdir("plates")) + 1}.jpg', plate_img)
            print(f"Placa original salva em: plates/placa_original_{len(os.listdir('plates'))}.jpg")

            # Salvar a imagem processada na pasta 'output'
            cv2.imwrite(f'output/placa_processada_{len(os.listdir("output")) + 1}.jpg', processed_plate)
            print(f"Placa processada salva em: output/placa_processada_{len(os.listdir('output'))}.jpg")

            # Visualizar as imagens
            visualize_images(image, plate_img, processed_plate)

            return processed_plate, box

    # Se não houver detecção
    print("Nenhuma placa detectada.")
    return None, None
