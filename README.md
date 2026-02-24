# 🥔 Potato Leaf Disease Detection

Application de détection automatique des maladies des feuilles de pomme de terre basée sur le Deep Learning. L'utilisateur soumet une photo de feuille et le modèle identifie si elle est saine ou atteinte d'une maladie.

---

## 🎯 Objectif

Aider les agriculteurs à détecter rapidement les maladies foliaires de la pomme de terre grâce à une interface simple et accessible, en utilisant un modèle de classification d'images entraîné sur le dataset **PlantVillage**.

---

## 🦠 Classes détectées

| Classe | Description |
|---|---|
| **Healthy** | Feuille saine, aucune maladie détectée |
| **Early Blight** | Mildiou précoce (*Alternaria solani*) — taches brunes circulaires |
| **Late Blight** | Mildiou tardif (*Phytophthora infestans*) — lésions sombres irrégulières, très contagieux |

---

## 🗂️ Structure du projet

```
potato_leaf_diseases-/
│
├── app.py                          # Application Streamlit
├── code_1.ipynb                    # Notebook d'exploration et d'entraînement
├── code_PLD_PLantVillage           # Script d'entraînement avec fine-tuning
├── potato_disease_model.h5         # Modèle entraîné (TensorFlow/Keras)
├── requirements.txt                # Dépendances Python
│
├── Training/                       # Images d'entraînement
├── Validation/                     # Images de validation
└── Testing/                        # Images de test
```

---

## 🧠 Modèle

- **Architecture** : CNN pré-entraîné avec fine-tuning (Transfer Learning)
- **Dataset** : [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) — Kaggle
- **Taille d'entrée** : 150 × 150 pixels (RGB)
- **Classes** : 3 (Healthy, Early Blight, Late Blight)
- **Augmentation de données** : zoom, cisaillement, retournement horizontal
- **Optimiseur** : Adam (lr=0.0001)
- **Loss** : Categorical Crossentropy
- **Callbacks** : EarlyStopping + ModelCheckpoint

---

## 🚀 Lancer l'application

### 1. Cloner le repo

```bash
git clone https://github.com/Gblack98/potato_leaf_diseases-.git
cd potato_leaf_diseases-
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv
source venv/bin/activate  # Windows : venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install streamlit tensorflow pillow numpy plotly
```

### 4. Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvre sur `http://localhost:8501`

---

## 🖥️ Utilisation

1. Ouvrir l'application dans le navigateur
2. Charger une image de feuille de pomme de terre (JPG, PNG, WEBP, TIFF, BMP)
3. Le modèle analyse l'image et affiche :
   - La **classe prédite** (Healthy / Early Blight / Late Blight)
   - Un **graphique des probabilités** par classe

---

## 📦 Technologies utilisées

| Outil | Rôle |
|---|---|
| Python | Langage principal |
| TensorFlow / Keras | Entraînement et inférence du modèle |
| Streamlit | Interface web interactive |
| Pillow | Traitement d'images |
| NumPy | Calculs matriciels |
| Plotly | Visualisation des probabilités |
| KaggleHub | Téléchargement du dataset |

---

## 👤 Auteur

**Gblack98** — [github.com/Gblack98](https://github.com/Gblack98)
