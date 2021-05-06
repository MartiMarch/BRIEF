"""
    Algoritmo BRIEF.
"""
import cv2

# Cámara utilizada
camara = cv2.VideoCapture(2, cv2.CAP_DSHOW)

# Se crea el detector de esquinas
detectorEsquinas = cv2.xfeatures2d.StarDetector_create()

# Se crea el objeto BRIEF
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

while True:

    # Se extrae una imagen
    _, imagen = camara.read()

    # Se encuentran los puntos clave
    puntosClave = detectorEsquinas.detect(imagen, None)

    # BRIEF procesa los puntos clave
    puntosClave, descriptor = brief.compute(imagen, puntosClave)

    # Se imprime el contenido del descriptor
    print(descriptor)

    # Se muestran los puntos clave
    cv2.imshow("BRIEF", imagen)

    # Pulsar 'q' para salir
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break