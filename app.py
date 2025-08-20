import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Configuración inicial
# -------------------------------
st.set_page_config(page_title="ML Demo App", layout="wide")
st.title("🤖 Demo: Modelos Supervisados en Streamlit")
st.write("Puedes subir tu propio CSV o usar datos simulados (300 muestras, 6 columnas).")

# -------------------------------
# 2. Subida de dataset
# -------------------------------
uploaded_file = st.file_uploader("📂 Sube un archivo CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset cargado correctamente.")
else:
    # Dataset simulado por defecto
    np.random.seed(42)
    n_samples = 300
    X = np.random.randn(n_samples, 6)
    y = (X[:, 0] + X[:, 1]*0.5 + np.random.randn(n_samples)*0.3 > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(1, 7)])
    df["target"] = y
    st.info("⚠️ No subiste un archivo. Se está usando un dataset simulado.")

# -------------------------------
# 3. Vista previa de datos
# -------------------------------
st.subheader("📋 Vista previa de los datos")
st.dataframe(df.head())

# -------------------------------
# 4. Selección de variables
# -------------------------------
if "target" in df.columns:
    target_col = st.selectbox("Selecciona la variable objetivo (target):", df.columns, index=len(df.columns)-1)
    feature_cols = st.multiselect("Selecciona las variables predictoras (features):", df.columns.drop(target_col), default=list(df.columns.drop(target_col)))
else:
    st.error("❌ El dataset debe contener una columna llamada 'target' o seleccionar una manualmente.")
    st.stop()

X = df[feature_cols]
y = df[target_col]

# -------------------------------
# 5. Selección de modelo
# -------------------------------
st.subheader("⚙️ Configuración del modelo")
model_name = st.radio("Elige el modelo supervisado:", ["Logistic Regression", "Decision Tree", "Random Forest"])

if model_name == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
elif model_name == "Decision Tree":
    model = DecisionTreeClassifier()
else:
    model = RandomForestClassifier()

# -------------------------------
# 6. Entrenamiento
# -------------------------------
test_size = st.slider("Proporción para test (%)", 10, 50, 20, step=5) / 100
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# -------------------------------
# 7. Resultados
# -------------------------------
st.subheader("📊 Resultados del modelo")
acc = accuracy_score(y_test, y_pred)
st.write(f"**Exactitud (accuracy):** {acc:.2f}")

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y), ax=ax)
ax.set_xlabel("Predicción")
ax.set_ylabel("Real")
st.pyplot(fig)

# Reporte de clasificación
st.text("Reporte de Clasificación:")
st.text(classification_report(y_test, y_pred))

# -------------------------------
# 8. Predicción manual
# -------------------------------
st.subheader("🔮 Haz una predicción manual")

manual_inputs = []
for col in feature_cols:
    val = st.number_input(f"Ingresar valor para {col}", value=0.0)
    manual_inputs.append(val)

if st.button("Predecir con valores manuales"):
    pred = model.predict([manual_inputs])[0]
    st.success(f"✅ El modelo predice: **{pred}**")
