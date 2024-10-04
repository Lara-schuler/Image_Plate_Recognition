import cv2

def load_yolo(model_cfg, model_weights):
    try:
        net = cv2.dnn.readNet(model_weights, model_cfg)
        print("Modelo YOLO carregado com sucesso.")
    except Exception as e:
        print(f"Erro ao carregar os arquivos de configuração do YOLO: {e}")
        return None, None

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers
