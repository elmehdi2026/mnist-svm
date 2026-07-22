import streamlit as st
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.datasets import fetch_openml

st.set_page_config(page_title="TP-SVM MNIST", page_icon="🎯", layout="centered")
st.header("EL MEHDI - IAENG")
st.title("🎯 Détecteur de '0' (One-Class SVM)")

# 1. Charger les données MNIST (Limité à 2500 images pour la légèreté)[cite: 5]
@st.cache_data
def load_data():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    return mnist.data[:2500] / 255.0, mnist.target[:2500]

with st.spinner("Chargement de MNIST..."):
    X, y = load_data()

# 2. Entraîner le modèle UNIQUEMENT sur les "0" (limité à 300 exemples pour la vitesse)[cite: 5]
@st.cache_resource
def train_model(X, y):
    X_zeros = X[y == '0'][:300] 
    model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
    model.fit(X_zeros)
    return model

with st.spinner("Entraînement du modèle One-Class SVM..."):
    model = train_model(X, y)

st.success("Modèle entraîné sur les '0' avec succès !")

# 3. Interface de test[cite: 5]
st.write("### 🔍 Tester le modèle de manière interactive")
chiffre = st.selectbox("Chiffre à tester :", list(range(10)))

if st.button("Tester une image aléatoire de ce chiffre"):
    indices = np.where(y == str(chiffre))[0]
    
    if len(indices) > 0:
        idx = np.random.choice(indices)
        img = X[idx]
        
        st.image(img.reshape(28, 28), width=150)
        
        pred = model.predict([img])
        
        st.write("---")
        if pred[0] == 1:
            st.write("Résultat : **C'est un 0 !**")
        else:
            st.write("Résultat : **Ce n'est PAS un 0 !**")
    else:
        st.warning("Aucune image trouvée pour ce chiffre dans l'échantillon.")
