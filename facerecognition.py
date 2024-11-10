import cv2
import os
from PIL import Image

import streamlit as st

face_cascade = cv2.CascadeClassifier('D:\majed\Data Science\deployment\checkpoint\haarcascade_frontalface_default.xml')

directory = r'D:\\majed\\Data Science\\deployment\\checkpoint\\FaceDetection'

os.chdir(directory)

def detect_faces() :

    # Initialiser la webcam
    caption=st.button("save image")
    cap = cv2.VideoCapture(0)
    face=cv2.imread(directory+'\\default.jpg')
    while True :

        # Lire les images de la webcam

        ret, frame = cap.read()

        # Convertit les images en niveaux de gris

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Détecter les visages à l'aide du classificateur de cascade de visages

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Dessine des rectangles autour des visages détectés

        for (x, y, w, h) in faces :

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Afficher les images

        cv2.imshow('Face Detection using Viola-Jones Algorithm', frame)

        # Sortir de la boucle lorsque 'q' est pressé

        
        if caption:
            face=frame 
            cv2.imwrite('face.jpg',face)
            caption=st.button("save image")
        
        st.image(face, caption='Voici la detection votre visage', use_column_width=True) 
        
        if cv2.waitKey(1) & 0xFF == ord('q') :

            break

    # Libère la webcam et ferme toutes les fenêtres

    cap.release()

    cv2.destroyAllWindows()


def app() :

    st.title("Face Detection using Viola-Jones Algorithm")

    st.write("Appuyez sur le bouton ci-dessous pour commencer à détecter des visages à partir de votre webcam")

    st.info("If want exit, just click 'q' ")

    # Ajouter un bouton pour commencer à détecter les visages

    
    if st.button("Detect Faces"):

        # Appeler la fonction detect_faces

        
       detect_faces()
 
 

if __name__ == "__main__" :

    app()


