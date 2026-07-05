import streamlit as st
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.datasets import fetch_openml

st.header("EL MEHDI - IAENG")
st.title("Détecteur de '0' (One-Class SVM)")

# 1. Charger les données MNIST
@st.cache_data
def load_data():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    return mnist.data / 255.0, mnist.target

st.write("Chargement de MNIST...")
X, y = load_data()

# 2. Entraîner le modèle UNIQUEMENT sur les "0"
@st.cache_resource
def train_model(X, y):
    X_zeros = X[y == '0']
    model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
    model.fit(X_zeros)
    return model

model = train_model(X, y)
st.success("Modèle entraîné sur les '0' !")

# 3. Interface de test
chiffre = st.selectbox("Chiffre à tester :", list(range(10)))
if st.button("Tester une image"):
    indices = np.where(y == str(chiffre))[0]
    idx = np.random.choice(indices)
    img = X[idx]
    
    st.image(img.reshape(28, 28), width=150)
    
    pred = model.predict([img])
    if pred[0] == 1:
        st.success("🎯 Résultat : C'est un 0 !")
    else:
        st.error("❌ Résultat : Ce n'est PAS un 0 !")
