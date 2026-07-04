# Plan de Trabajo Implementado: Proyecto de Análisis Eléctrico

Este documento detalla el flujo de trabajo metodológico que seguimos para analizar el dataset de generación eléctrica (`2107_electrical_data.csv`). Todo el desarrollo técnico fue plasmado de manera modular en el archivo `analisis_electrico.ipynb`.

## 1. Etapa 1: Comprensión del Problema
*   **Objetivo:** Explorar la estructura general de los datos para definir una pregunta de negocio avanzada, evitando los enfoques triviales.
*   **Acciones realizadas:**
    *   Carga inicial de una muestra de datos usando `nrows` para no saturar la memoria.
    *   Inspección de las columnas (120 variables de sensores de 24 inversores).
*   **Definición del caso de uso:** Se planteó el **Diagnóstico de Estrés Térmico (Thermal Derating)**. En lugar de solo buscar los que menos energía generan, decidimos construir un "sensor virtual" matemático para detectar caídas anómalas de eficiencia debido al calor del mediodía.

## 2. Etapa 2: Análisis Exploratorio de Datos (EDA)
*   **Objetivo:** Comprender las dimensiones, tipos de datos y estadísticas descriptivas de las variables eléctricas.
*   **Acciones realizadas:**
    *   Uso de `.info()` para detectar tipos de datos (como fechas no parseadas) y `.describe()` para estadísticas de la métrica cruda.
    *   Graficado de curvas de tendencia temporal (`lineplot`) sobre los primeros días operativos para identificar visualmente el "valle" del mediodía que acusa el estrés térmico en el inversor.

## 3. Etapa 3: Preparación de Datos (A Escala)
*   **Objetivo:** Limpiar y transformar los datos de los 24 inversores para prepararlos para el modelado, manejando problemas del mundo real.
*   **Acciones realizadas:**
    *   **Ingeniería de Características:** Creación de la variable `dc_power` (Voltaje * Corriente) y la métrica de **Eficiencia** ajustando las unidades (Multiplicando KiloWatts a Watts).
    *   **Tratamiento de Nulos y Ruido:** Filtrado de horarios sin luz solar (`dc_power > 10`) para evitar divisiones entre cero y datos nulos.
    *   **Tolerancia a fallos:** Implementación de un bloque `try-except` que reveló que los inversores 05 y 06 carecían del sensor de voltaje.
    *   **Outliers:** Acotamiento de la eficiencia con `np.clip` (máximo 100%) respetando las leyes de la física.
    *   **Validación Cuantitativa:** Creación de una tabla ordenada por la mediana (`.median()`) para respaldar las deducciones de los gráficos densos (Boxplot), demostrando que el Inversor 22 era el más estable y el Inversor 06 el más crítico.

## 4. Etapa 4: Modelado y Algoritmia
*   **Objetivo:** Aplicar las tres técnicas principales de Machine Learning requeridas por la rúbrica del proyecto.
*   **Modelos desarrollados:**
    1.  **Aprendizaje No Supervisado (K-Means):** Se agruparon los inversores en 3 clústers (Óptimo, Advertencia, Crítico) basándonos en su mediana de rendimiento y su volatilidad (desviación estándar). Requiere escalado de datos (`StandardScaler`).
    2.  **Aprendizaje Supervisado (Regresión Lineal):** Se entrenó un modelo para predecir la potencia de salida (AC Power) del Inversor 22 basándose en su entrada DC. Logramos un $R^2$ superior al 99%.
    3.  **Forecasting (Series de Tiempo):** Se emplearon variables rezagadas (`lags` de 1h y 24h) para predecir la generación de energía total del parque en el corto plazo.

## 5. Etapa 5: Generación de Evidencias (Reporte)
*   **Objetivo:** Guardar todas las pruebas visuales y tabulares necesarias para estructurar el dictamen evaluativo.
*   **Evidencias generadas en la carpeta `/evidencias`:**
    *   `01_boxplot_eficiencia.png`
    *   `02_tabla_eficiencia_mediana.csv`
    *   `03_kmeans_clusters.png`
    *   `04_forecasting.png`
    *   `05_supervisado_real_vs_pred.png`

---
*Nota: Este plan fue implementado utilizando buenas prácticas de programación en Python, prevención de fuga de datos (Data Leakage) en series temporales y visualización analítica de datos.*
