import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Generación de datos simulados
# -------------------------------
np.random.seed(42)
n_samples = 300
X = np.random.randn(n_samples, 6)  # 6 columnas de features
y = (X[:, 0] + X[:, 1]*0.5 + np.random.randn(n_samples)*0.3 > 0).astype(int)  # target binario

df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(1, 7)])
df["target"] = y

# -------------------------------
# 2. Interfaz con Streamlit
# -------------------------------
st.title("📊 Demo: Modelo Supervisado con Streamlit")
st.write("Ejemplo con datos simulados (300 muestras, 6 features).")

st.subheader("Vista previa de los datos")
st.dataframe(df.head())

# -------------------------------
# 3. Entrenamiento del modelo
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df.drop("target", axis=1), df["target"], test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

# -------------------------------
# 4. Predicciones y métricas
# -------------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.subheader("Resultados del modelo")
st.write(f"**Exactitud (accuracy):** {acc:.2f}")

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1], ax=ax)
ax.set_xlabel("Predicción")
ax.set_ylabel("Real")
st.pyplot(fig)

# -------------------------------
# 5. Predicción con entrada manual
# -------------------------------
st.subheader("🔮 Haz una predicción manual")

inputs = []
for i in range(1, 7):
    val = st.number_input(f"Ingresar valor para feature_{i}", value=0.0)
    inputs.append(val)

if st.button("Predecir"):
    pred = model.predict([inputs])[0]
    st.success(f"El modelo predice: **{pred}** (0 = Clase Negativa, 1 = Clase Positiva)")
