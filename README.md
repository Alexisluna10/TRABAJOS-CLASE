# Sistemas Inteligentes

Este repositorio contiene la implementación y el informe técnico de la Actividad 11 de la materia Sistemas Inteligentes I (Periodo Mayo–Agosto 2025) en la Universidad Politécnica Metropolitana de Hidalgo.

# 🔍 Descripción

Se desarrolla un modelo híbrido de redes neuronales que combina capas convolucionales (CNN) y secuenciales (LSTM) para clasificar las 10 categorías del conjunto de datos CIFAR‑10. El objetivo es aprovechar la extracción de características espaciales de la CNN y la modelación de patrones temporales de la LSTM para mejorar la precisión de clasificación.


# 📁 Estructura de proyecto

```

/
├── TAREA-11-SISTEMAS-INTELIGENTES-I-MAYO-AGOSTO-2025-DOMINGUEZ.M-LUNA.S-MARQUEZ.R.docx  # Informe en Word de la Tarea 11
├── código/                                                                              # Código fuente de la Tarea 11
│   ├── prueba2.py                                                                       # Script de entrenamiento                                                       
└── README.md 

```

# ⚙️ RequisitosPython 3.8 o superior

- TensorFlow 2.x
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- Opcional: GPU compatible (se configuró para CPU AMD Ryzen 7 5700U)

# Dependencias

```bash
pip install tensorflow numpy scikit-learn matplotlib seaborn
```

# 📊 Resultados

- Precisión de prueba y pérdida al finalizar el entrenamiento.
- Curvas de aprendizaje, matriz de confusión, reporte de clasificación y distribución de confianza.
- Estadísticas de los pesos sinápticos y polarizaciones para inspección y reutilización.

# Autores

- Dominguez Martinez Fernando
- Luna Santillán Alexis
- Márquez Ramirez Dayana
