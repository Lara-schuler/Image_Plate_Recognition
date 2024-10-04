import easyocr

def recognize_plate(plate_image):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(plate_image)

    if result:
        return result[0][-2]  # Retorna o texto detectado
    return None
