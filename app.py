import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Configuración inicial
# -------------------------------
st.set_page_config(page_title="Árbol de Decisión - Análisis de Datos", layout="wide")
st.title("🌱 Análisis de Datos con Árbol de Decisión")
st.write("Sube un archivo CSV (ej: datos de agricultura) o usa datos simulados.")

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
target_col = st.selectbox("Selecciona la variable objetivo (target):", df.columns, index=len(df.columns)-1)
feature_cols = st.multiselect("Selecciona las variables predictoras (features):", df.columns.drop(target_col), default=list(df.columns.drop(target_col)))

X = df[feature_cols]
y = df[target_col]

# -------------------------------
# 5. Configuración del Árbol
# -------------------------------
st.subheader("⚙️ Configuración del Árbol de Decisión")
max_depth = st.slider("Profundidad máxima del árbol", 1, 15, 5)
criterion = st.radio("Criterio de división:", ["gini", "entropy"], index=0)

model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=42)

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
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=np.unique(y), yticklabels=np.unique(y), ax=ax)
ax.set_xlabel("Predicción")
ax.set_ylabel("Real")
st.pyplot(fig)

# Reporte de clasificación
st.text("Reporte de Clasificación:")
st.text(classification_report(y_test, y_pred))

# Importancia de variables
st.subheader("📌 Importancia de las variables")
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
st.bar_chart(importances)

# Visualización del árbol
st.subheader("🌳 Visualización del Árbol de Decisión")
fig, ax = plt.subplots(figsize=(12, 6))
plot_tree(model, feature_names=feature_cols, class_names=[str(c) for c in np.unique(y)], filled=True, ax=ax)
st.pyplot(fig)

# -------------------------------
# 8. Predicción manual
# -------------------------------
st.subheader("🔮 Haz una predicción manual")

manual_inputs = []
for col in feature_cols:
    if pd.api.types.is_numeric_dtype(df[col]):
        val = st.number_input(f"Ingresar valor para {col}", value=float(df[col].mean()))
    else:
        opciones = df[col].unique().tolist()
        val = st.selectbox(f"Selecciona valor para {col}", opciones)
    manual_inputs.append(val)

if st.button("Predecir con valores manuales"):
    manual_df = pd.DataFrame([manual_inputs], columns=feature_cols)
    pred = model.predict(manual_df)[0]
    st.success(f"✅ El árbol predice: **{pred}**")
