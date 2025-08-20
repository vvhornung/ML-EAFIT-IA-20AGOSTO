# ML-EAFIT-IA-20AGOSTO

---

# 📘 README - Modelos de Machine Learning: Supervisados y No Supervisados

## 🔹 ¿Qué es el Machine Learning?

El **Machine Learning (ML)** es un subcampo de la Inteligencia Artificial que permite a los sistemas **aprender patrones a partir de datos** para realizar predicciones, clasificaciones o descubrimientos sin estar explícitamente programados.

Existen dos grandes enfoques: **supervisado** y **no supervisado**.

---

## 🔹 1. Modelos Supervisados

En el **aprendizaje supervisado**, los datos de entrenamiento están **etiquetados** (es decir, conocemos la variable de salida).
El objetivo es **aprender la relación entre las variables de entrada (features) y la salida esperada (label)**.

### ✅ Usos comunes

* Predicciones numéricas (ej. precio de una casa).
* Clasificación de categorías (ej. spam/no spam en correos).

### 📌 Ejemplos de algoritmos supervisados:

1. **Regresión Lineal** – Para predecir valores numéricos.
2. **Regresión Logística** – Para clasificación binaria.
3. **Árboles de Decisión** – Clasificación y regresión interpretables.
4. **Random Forest** – Conjunto de árboles para mayor robustez.
5. **Support Vector Machines (SVM)** – Clasificación con márgenes óptimos.
6. **Redes Neuronales Artificiales (ANN)** – Modelos más complejos para grandes volúmenes de datos.

### 📊 Ejemplo práctico

* **Problema:** Predecir si un paciente tiene diabetes.
* **Entrada (X):** Edad, IMC, nivel de glucosa.
* **Salida (y):** 0 = No tiene diabetes, 1 = Tiene diabetes.

---

## 🔹 2. Modelos No Supervisados

En el **aprendizaje no supervisado**, los datos **no tienen etiquetas**.
El objetivo es **descubrir patrones ocultos, estructuras o agrupaciones en los datos**.

### ✅ Usos comunes

* Segmentación de clientes en marketing.
* Detección de anomalías en fraudes bancarios.
* Reducción de dimensionalidad para visualización de datos.

### 📌 Ejemplos de algoritmos no supervisados:

1. **K-Means** – Agrupamiento de datos en k clústeres.
2. **Clustering Jerárquico** – Construye jerarquías de agrupamientos.
3. **DBSCAN** – Descubre clústeres basados en densidad.
4. **PCA (Análisis de Componentes Principales)** – Reducción de dimensionalidad.
5. **Autoencoders** – Redes neuronales para codificar/descodificar información.

### 📊 Ejemplo práctico

* **Problema:** Segmentar clientes de un supermercado.
* **Entrada (X):** Frecuencia de compras, monto gastado, categorías compradas.
* **Salida (y):** No existe (el algoritmo descubre los grupos).

---

## 🔹 3. Diferencias Clave

| Característica      | Supervisado 🧑‍🏫                | No Supervisado 🕵️                         |
| ------------------- | -------------------------------- | ------------------------------------------ |
| **Datos de salida** | Con etiquetas                    | Sin etiquetas                              |
| **Objetivo**        | Predecir valores o clasificar    | Encontrar patrones ocultos                 |
| **Ejemplo típico**  | Spam / No spam                   | Agrupar clientes                           |
| **Evaluación**      | Precisión, recall, MSE, accuracy | Coeficiente de silueta, varianza explicada |

---

## 🔹 4. Conclusiones

* Usa **supervisado** cuando tengas **datos etiquetados** y quieras predecir algo.
* Usa **no supervisado** cuando quieras **explorar patrones** sin conocer previamente la respuesta.
* En la práctica, muchas soluciones combinan ambos enfoques (**aprendizaje semi-supervisado** o **auto-supervisado**).

---

¿Quieres que te lo organice también en **formato Markdown con ejemplos de código en Python** (sklearn) para que sea un README práctico en GitHub? 🚀
