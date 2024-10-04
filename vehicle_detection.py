import cv2
import numpy as np
from yolo_utils import load_yolo

def resize_image_if_needed(image, min_size=800):
    height, width, _ = image.shape
    if height < min_size or width < min_size:
        scale_factor = min_size / min(height, width)
        new_dimensions = (int(width * scale_factor), int(height * scale_factor))
        resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_LINEAR)
        print(f"Imagem redimensionada para {new_dimensions}")
        return resized_image
    return image

def detect_vehicle(image, model_cfg, model_weights, confidence_threshold=0.3, nms_threshold=0.5):
    # Redimensionar a imagem se necessário
    image = resize_image_if_needed(image)

    # Carregar o modelo YOLO
    net, output_layers = load_yolo(model_cfg, model_weights)

    if net is None or output_layers is None:
        print("Falha ao carregar o modelo YOLO.")
        return None, None

    # Extrair altura e largura da imagem
    height, width, _ = image.shape

    # Pré-processamento da imagem para o YOLO
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # IDs das classes que queremos: Carro (2), Moto (3), Ônibus (5), Caminhão (7)
    wanted_classes = [2, 3, 5, 7]

    # Processamento da saída
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold:
                if class_id in wanted_classes:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    # Aplica Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    # Verifique se `indices` não está vazio e é uma lista
    if len(indices) > 0:
        # `indices` pode ser uma lista de listas ou um array 2D
        for i in indices.flatten():  # Use `flatten()` para lidar com diferentes formatos
            box = boxes[i]
            x, y, w, h = box
            # Recortar a imagem do veículo
            vehicle_img = image[y:y + h, x:x + w]
            return vehicle_img, box

    # Se não houver detecção
    print("Nenhum veículo detectado.")
    return None, None
