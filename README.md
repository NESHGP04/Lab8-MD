# Lab 8 — Máquinas Vectoriales de Soporte (SVM y SVR)

**Curso:** Minería de Datos
**Dataset:** Airbnb listings (`listings.RData`) — 76,246 propiedades, 80 variables
**Problemas abordados:**
- **Clasificación multiclase** — predecir si una propiedad es `Económico` (≤$120), `Intermedio` ($121–$326) o `Caro` (>$326).
- **Regresión** — predecir el precio exacto de la propiedad.

---

## Contenido del notebook (`main.ipynb`)

### Inciso 1 — Conjuntos de entrenamiento y prueba
Limpieza de la columna `price` (remover símbolos de dólar y comas, convertir a numérico). División 70/30 estratificada con `random_state=42` usando 5 features predictoras: `accommodates`, `bathrooms`, `bedrooms`, `beds`, `review_scores_rating`. Imputación con la media para valores faltantes.

### Inciso 2 — Exploración y transformación
Escalado con `StandardScaler` aplicado a `X_train` y `X_test`. SVM requiere features en la misma escala porque calcula distancias en el espacio de features.

### Inciso 3 — Variable respuesta
`price_category` con 3 clases: `Económico` (25.3%), `Intermedio` (49.7%), `Caro` (24.9%).

### Inciso 4 — Modelos SVM con diferentes kernels
Tres modelos entrenados:
- **RBF** (`C=1`): accuracy = **0.6164**
- **Lineal** (`C=1`): accuracy = **0.6058**
- **Polinomial** (`degree=3, C=1`): accuracy = **0.5785**

Tuneo adicional de `gamma` para RBF: el mejor valor fue `gamma=1` (accuracy 0.6258).

### Inciso 5 — Predicción
Predicciones generadas para los tres modelos sobre `X_test_scaled`.

### Inciso 6 — Matrices de confusión
Heatmaps de confusion matrix para los 3 kernels. Los tres modelos confunden principalmente clases adyacentes (`Económico`↔`Intermedio` e `Intermedio`↔`Caro`).

### Inciso 7 — Análisis de sobreajuste / desajuste
Comparación de accuracy train vs test. Los gaps son ≤1%, descartando sobreajuste pero confirmando **desajuste** (todas las accuracies en torno a 58–62%). Curvas de validación variando `C` y `degree`.

### Inciso 8 — Comparación de efectividad, tiempo y errores
| Kernel | Accuracy | Tiempo Fit |
|--------|----------|------------|
| Lineal | 0.606 | 37.2 s |
| RBF | 0.616 | 45.9 s |
| Polinomial | 0.579 | 158.7 s |

`classification_report` por modelo. **Mejor SVM elegido: RBF con C=1**.

### Inciso 9 — Comparación con otros algoritmos de clasificación
Comparación contra Árbol de Decisión, Random Forest, Naive Bayes, KNN y Regresión Logística:

| # | Modelo | Accuracy | F1 | Tiempo Fit | Tiempo Pred |
|---|--------|----------|-----|------------|-------------|
| 1 | SVM (RBF, C=1) | 0.6164 | 0.6071 | 46.39 s | 17.09 s |
| 2 | Random Forest | 0.6136 | 0.6104 | 0.33 s | 0.04 s |
| 3 | Regresión Logística | 0.6096 | 0.6023 | 0.11 s | 0.001 s |
| 4 | Árbol de Decisión | 0.6028 | 0.6009 | 0.08 s | 0.003 s |
| 5 | KNN (k=5) | 0.5887 | 0.5879 | 0.06 s | 0.49 s |
| 6 | Naive Bayes | 0.5298 | 0.5270 | 0.03 s | 0.002 s |

**Veredicto:** Random Forest gana en relación costo/desempeño — pierde solo 0.003 puntos de accuracy contra SVM pero corre ~150× más rápido.

### Inciso 10 — Tabla comparativa de sobreajuste
Análisis del gap accuracy train − test para los 6 modelos:

| Modelo | Acc Train | Acc Test | Gap | Diagnóstico |
|--------|-----------|----------|-----|-------------|
| Árbol de Decisión | 0.7067 | 0.6028 | +10.4% | Sobreajuste |
| Random Forest | 0.7067 | 0.6136 | +9.3% | Sobreajuste |
| KNN (k=5) | 0.6383 | 0.5887 | +5.0% | Sobreajuste leve |
| Naive Bayes | 0.5308 | 0.5298 | +0.1% | Desajuste |
| SVM (RBF, C=1) | 0.6120 | 0.6164 | −0.4% | Desajuste |
| Regresión Logística | 0.6024 | 0.6096 | −0.7% | Desajuste |

**Parámetros usados para detectar sobreajuste:** accuracy train vs test, F1 train vs test, gap (con umbrales 2% y 5%), validación cruzada k-fold y curvas de aprendizaje.

### Inciso 11 — Modelo de regresión SVR
Regresión sobre la variable continua `price` con cap p95 (~$1,034) para filtrar outliers extremos.

- **Tuneo:** `GridSearchCV` con cv=3 sobre subsample de 5,000 filas (SVR es O(n²–n³)).
- **Mejor configuración:** `kernel='linear', C=10, gamma='scale'`.
- **Métricas en test:** RMSE = **155.32**, MAE = **100.27**, R² = **0.2779**.
- **Diagnóstico:** modelo bien ajustado (gap R² = 0.003), pero con desajuste (R² bajo).

### Inciso 12 — Comparación con otros modelos de regresión
Comparación contra Regresión Lineal, Árbol de Regresión, Random Forest Regressor y KNN Regressor:

| # | Modelo | RMSE Test | MAE Test | R² Test | Gap R² | Tiempo Fit |
|---|--------|-----------|----------|---------|--------|------------|
| 1 | Random Forest | 147.97 | 101.19 | 0.3446 | 0.212 | 0.24 s |
| 2 | Regresión Lineal | 151.35 | 105.17 | 0.3143 | 0.003 | 0.01 s |
| 3 | SVR (mejor) | 155.32 | 100.27 | 0.2779 | 0.003 | 50.26 s |
| 4 | KNN (k=5) | 156.57 | 108.49 | 0.2662 | 0.110 | 0.03 s |
| 5 | Árbol de Regresión | 160.94 | 107.29 | 0.2246 | 0.352 | 0.04 s |

**Veredicto:** Random Forest gana también en regresión (mejor R² y RMSE, ~210× más rápido que SVR). Naive Bayes se omite por ser un clasificador y no aplicar a regresión continua.

---

## Variables predictoras utilizadas

| Variable | Descripción |
|----------|-------------|
| `accommodates` | Capacidad máxima de huéspedes |
| `bathrooms` | Número de baños |
| `bedrooms` | Número de habitaciones |
| `beds` | Número de camas |
| `review_scores_rating` | Puntuación promedio de reviews |

---

## Hallazgos principales

1. **SVM RBF** logra la mejor accuracy de clasificación (0.6164) por margen mínimo, pero a un costo computacional 150× mayor que Random Forest.
2. **Random Forest** es la opción más práctica tanto en clasificación como en regresión: mejor o equivalente desempeño con tiempo de procesamiento despreciable.
3. **El kernel lineal ganó** en SVR sobre el lineal y RBF — con solo 5 features numéricas no hay relaciones no lineales fuertes que explotar.
4. **Cuello de botella:** las 5 features no son suficientes para superar ~62% accuracy en clasificación o ~0.35 R² en regresión. Para mejorar significativamente sería necesario incluir `neighbourhood`, `room_type`, `property_type` y amenities.
5. **Diagnóstico de sobreajuste claro:** Árbol de Decisión y Random Forest sobreajustan (gap ~10%), mientras que SVM, Regresión Logística y Naive Bayes están desajustados (gap ≤0).

---

## Estructura del proyecto

```
Lab8-MD/
├── main.ipynb              # Notebook principal con incisos 1-12
├── AnalisisExp.ipynb       # Análisis exploratorio inicial
├── listings.RData          # Dataset Airbnb (76,246 filas)
├── ejemploClase/           # Material de referencia del curso
│   └── Ejemplo de SVM.ipynb
├── data/                   # Datos auxiliares
├── requirements.txt        # Dependencias Python
└── README.md               # Este archivo
```

## Dependencias

```
pandas, numpy, scikit-learn, seaborn, matplotlib, pyreadr
```

Instalación:
```bash
pip install -r requirements.txt
```
