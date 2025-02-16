import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import plotly.express as px

# Configuration de la page (titre, icône, layout)
st.set_page_config(
    page_title="Détection Maladies Pomme de Terre",
    page_icon="🥔",
    layout="centered"
)

# Titre de l'appli
st.title("Détection de maladies de la pomme de terre🌱")
st.write("Téléverse une image de feuille et laisse l'IA faire son taff !")

# Charger le modèle
model = tf.keras.models.load_model("potato_disease_model.h5")

# Classes dans l'ordre du modèle
class_names = ['Late_Blight', 'Healthy', 'Early_Blight']

# Uploader une image
uploaded_file = st.file_uploader(
    "Choisis ton image :", 
    type=["jpg", "jpeg", "png", "webp", "tiff", "bmp"]
)

# Animation d'accueil (pour le fun) : un petit effet neige
if uploaded_file is not None:
    # Ouvrir l'image en mode RGB pour éviter les soucis de canal alpha
    image = Image.open(uploaded_file).convert("RGB")
    
    # Afficher l'image chargée
    st.image(image, caption='Image chargée', use_container_width=True)

    # Afficher un spinner pendant la prédiction
    with st.spinner("Analyse en cours... 🤖"):
        # Redimensionner l'image comme dans l'entraînement
        img = image.resize((150, 150))
        # Convertir en tableau NumPy et normaliser
        img_array = np.array(img) / 255.0
        # Ajouter une dimension pour le batch
        img_array = np.expand_dims(img_array, axis=0)

        # Faire la prédiction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_probabilities = predictions[0]

    # Affichage du résultat principal
    st.success(f"**Résultat de la prédiction** : {class_names[predicted_class]}")


    # Créer un graphique en barres pour afficher la probabilité de chaque classe
    fig = px.bar(
        x=class_names,
        y=predicted_probabilities,
        color=class_names,
        labels={'x': 'Classes', 'y': 'Probabilité'},
        title="Probabilités par classe"
    )
    # Afficher le graphique
    st.plotly_chart(fig)

    st.write(" Si ta feuille est Late Blight, prépare-toi à sauver ton champ ! 🚀")
else:
    st.info("Téléverse une image pour lancer la détection !")
