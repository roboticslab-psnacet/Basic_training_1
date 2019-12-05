# Importar las bibliotecas necesarias
from collections import deque
import numpy as np
import argparse
import imutils
import cv2

# construye el argumento parse
ap = argparse.ArgumentParser()
args = vars(ap.parse_args())

lower = {'red': (166, 84, 141), 'green': (66, 122, 129), 'blue': (97, 100, 117), 'yellow': (23, 59, 119), 'orange': (0, 50, 80)}
upper = {'red': (186, 255, 255), 'green': (86, 255, 255), 'blue': (117, 255, 255), 'yellow': (54, 255, 255), 'orange': (20, 255, 255)}

colors = {'red': (0, 0, 255), 'green': (0, 255, 0), 'blue': (255, 0, 0), 'yellow': (0, 255, 217), 'orange': (0, 140, 255), 'brown': (165, 42, 42)}

camera = cv2.VideoCapture(0)

# Loop
while True:
    # Obtenemos cuadro por cuadro tomado, y lo convertimos a una formato
    # de color HSV, ya que se representa de una mejor manera con esto.
    (grabbed, frame) = camera.read()

    # Si estamos viendo un video y no tomamos un fotograma,
    # hemos llegado al final del video.
    if args.get("video") and not grabbed:
        break

    # Se crea un tamano distinto
    frame = imutils.resize(frame, width=600)

    # Se suaviza la imagen utilizando un filtro gaussiano, eliminando
    # cualquier presencia de ruido en el frame
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Se utiliza una serie de dilataciones y erosiones a la imagen para
  
    for key, value in upper.items():
        # Utiliza un kernel de 9x9
        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.inRange(hsv, lower[key], upper[key])

        # Reduce y luego expande: opening
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Expandey luego reduce: closing
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Encuentra el contorno de una imagen binaria e iniciliza el centro de la bola(x, y)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None

       
        if len(cnts) > 0:
            # Encuentra el contorno mas largo de la mascara
            # Luego lo usa para calcular el circulo y centroide minimo

            # cv2.contourArea calcula el area de un contorno
            c = max(cnts, key=cv2.contourArea)

            # cv2.minEnclosingCircle()
            ((x, y), radius) = cv2.minEnclosingCircle(c)

            # cv2.moments calcula todos los momentos hasta el tercer
            # orden de un poligono o una figura rasterizada
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] /
                                                    M["m00"]))

           
            if radius > 0.5:
                cv2.circle(frame, (int(x), int(y)), int(radius), colors[key], 2)
                cv2.putText(frame, key + " object", (int(x - radius), int(y - radius)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[key], 2)

        # Muestra el cuadro a nuestra ventana
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # Cuando 'q' es presionado, deten el ciclo for
        if key == ord("q"):
            break

            # Limpia la camara y cierra cualquier ventana abierta
            camera.release()
            cv2.destroyAllWindows()
