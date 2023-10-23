# Anthony RENARD 🦊-  23/10/2023
# https://www.linkedin.com/in/anthonyrenardfox/

# Détections d'objets en live par Yolov8n.pt

# Importation des blibliothèques utiles
import cv2
import supervision as sv
from ultralytics import YOLO



def main():
    capture = cv2.VideoCapture(0)  # Activation de la caméra

    model = YOLO("yolov8n.pt") #Chargement du modèle

    # Création des boites annotées lors de la détection
    boite_annotee= sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=1)

    while True:  # Création d'une boucle infinie pour faire de la détection sur la vidéo
        ret, frame = capture.read()  # Lecture d'une image de la webcam

        result = model(frame)[
            0
        ]  # utilisation du model pour détecter les objets sur une trame de la vidéo
        detections = sv.Detections.from_ultralytics(
            result
        )  # Création de l'image avec ses boites de détection et la probabilité associée sur chaque détection
        etiquettes = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _ in detections
        ]  # Ajout des étiquettes à chaque détection (en plus des problabilités)

        frame = boite_annotee.annotate(
            scene=frame, detections=detections, labels=etiquettes
        )  # Création de l'image annotée

        cv2.imshow("Detection des objets sur votre webcam par Anthony RENARD", frame)  # Affichage de l'image annotée

        if (
            cv2.waitKey(30) == 27
        ):  # Le code ASCII 27 correspond à la touche "escape" sur un clavier. On attend ici un appuie de 30ms sur l'image webcam pour stopper le processus et donc éteindre la webcam
            break


if __name__ == "__main__":
    main()
