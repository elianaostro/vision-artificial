# Visión Artificial

Este repositorio contiene una serie de tutoriales y trabajos prácticos centrados en los fundamentos y técnicas avanzadas de **Visión Artificial**, utilizando principalmente la librería **OpenCV** y **Python**.

## Contenidos

### 🧪 Trabajos Prácticos

  - **TP 0 - Umbralización y Reconocimiento de Regiones**: Experimentación con métodos de binarización, segmentación y algoritmos de componentes conexas para el reconocimiento de objetos por color y forma.

### 📚 Tutoriales

El repositorio está organizado en tutoriales progresivos que cubren desde operaciones básicas hasta geometría de múltiples vistas:

1.  **Procesamiento de Imágenes**: Operaciones básicas, manejo de canales (RGB, HSV), histogramas y transformaciones puntuales.
2.  **Filtros Espaciales y Detección de Bordes**: Aplicación de kernels de convolución, filtros gaussianos, detección de bordes y filtrado en el dominio espacial.
3.  **Features y Matching**: Detección de puntos clave (Keypoints) mediante descriptores invariantes locales (como SIFT) y técnicas de emparejamiento de características.
4.  **Homografías**: Transformaciones geométricas, estimación de matrices de homografía, algoritmos DLT y RANSAC para robustez ante outliers.
5.  **Calibración de Cámara**: Modelos de cámara (Pinhole), estimación de parámetros intrínsecos/extrínsecos y corrección de distorsión de lentes usando patrones de checkerboard.
6.  **Visión Estéreo**: Calibración y rectificación estéreo, generación de mapas de disparidad y reconstrucción de nubes de puntos 3D.

## Configuración y Ejecución

Los notebooks están diseñados para ejecutarse tanto en un entorno local como en **Google Colab**.

### Requisitos

  - Python 3.x
  - OpenCV (`cv2`)
  - NumPy
  - Matplotlib
  - [i308-utils](https://www.google.com/search?q=https://github.com/udesa-vision/i308-utils) (Paquete de utilidades para visualización)

### Instalación de utilidades

En cada notebook se puede instalar el paquete de herramientas necesarias mediante:

```bash
pip install git+https://github.com/udesa-vision/i308-utils.git
```

## Estructura del Proyecto

  - `tp0_umbralizacion_regiones/`: Recursos y código del TP inicial.
  - `tutorial_01/` a `tutorial_06/`: Notebooks interactivos y recursos (`res/`) para cada tema de la agenda.
