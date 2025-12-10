import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# ============================================
# DATASET
# ============================================

log_rho = np.array([
    -7.80, -7.77, -7.61, -7.58, -7.25, -7.01, -6.97, -6.96, -7.16, -7.23,
    -7.15, -6.84, -6.66, -7.35, -6.38,
    10.0, 10.3, 17.9, 12.0, 11.0, 16.0, 14.0, 14.0, 13.0, 14.0,
    10.0, 11.95, 12.0, 2.81, 9.0
])

thermal_k = np.array([
    429, 401, 318, 237, 173, 80, 72, 67, 91, 116,
    109, 54, 35, 156, 22,
    0.8, 1.0, 1.4, 1.5, 1.2, 0.25, 0.35, 0.19, 0.15, 0.17,
    0.05, 0.71, 0.35, 148, 0.23
])

labels = np.array([0]*15 + [1]*15)  # konduktor=0, isolator=1

X = np.column_stack([log_rho, thermal_k])
y = labels

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model ANN
model = MLPClassifier(hidden_layer_sizes=(10,5), activation="relu",
                      max_iter=2000, solver="adam", random_state=42)
model.fit(X_scaled, y)

# ============================================
# STREAMLIT WEB UI
# ============================================

st.title("🔬 Web ANN Material Classifier")
st.write("Aplikasi web untuk mengklasifikasikan material berdasarkan resistivitas & konduktivitas termal.")

log_input = st.number_input("Masukkan log10(resistivitas)", value=-7.5)
thermal_input = st.number_input("Masukkan konduktivitas termal (W/mK)", value=200.0)

if st.button("Prediksi Material"):
    X_new = np.array([[log_input, thermal_input]])
    X_new_scaled = scaler.transform(X_new)

    pred = model.predict(X_new_scaled)[0]
    conf = model.predict_proba(X_new_scaled)[0]

    jenis = "Konduktor" if pred == 0 else "Isolator"

    st.subheader("Hasil Prediksi")
    st.write(f"Jenis Material : **{jenis}**")
    st.write(f"Confidence : **{max(conf):.4f}**")

    # Plot Decision Boundary
    st.subheader("Grafik Decision Boundary ANN")

    x_min, x_max = X_scaled[:,0].min()-1, X_scaled[:,0].max()+1
    y_min, y_max = X_scaled[:,1].min()-1, X_scaled[:,1].max()+1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7,5))
    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(X_scaled[:,0], X_scaled[:,1], c=y, edgecolors='k')
    ax.set_xlabel("Fitur 1 (scaled)")
    ax.set_ylabel("Fitur 2 (scaled)")
    st.pyplot(fig)
