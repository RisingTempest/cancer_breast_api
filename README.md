# Proyecto: Contenerización de una API ML con Docker

## Descripción
Este proyecto desarrolla una **API REST** usando **Flask** que expone un modelo de clasificación de tumores de mama basado en el dataset **Breast Cancer** de scikit-learn.

El modelo entrenado es un **RandomForestClassifier** con búsqueda aleatoria de hiperparámetros (RandomizedSearchCV) para optimizar su desempeño.

La API permite recibir un conjunto de características mediante una solicitud **POST** y devuelve la predicción en formato JSON, indicando si el tumor es **Benigno** o **Maligno**.

Además, el proyecto está contenarizado con **Docker**, lo que permite ejecutar la API de manera aislada y reproducible en cualquier sistema que soporte Docker.

---

## Requisitos

```bash
Python ≥ 3.10.0  
```

**Librerías necesarias:**

```bash
- flask==3.1.2
- scikit-learn==1.7.0
- numpy==1.26.2
- joblib==1.3.2
- requests==2.31.0
```

**Entorno Docker:**

Para correr la API dentro de un contenedor:

- Tener Docker Desktop instalado (Windows, macOS o Linux)
  - En Windows, se recomienda versión AMD64 o ARM64 según tu procesador
- Habilitar WSL2 en Windows y descargar una distribución Linux (Ubuntu)
- Verificar que Docker se ejecuta correctamente (`docker version`)
- El puerto 5000 debe estar libre en tu máquina, ya que es el que se mapea para exponer la API.

## Estructura de Archivos

```bash
/Actividad1_GonzaloMoyano
├─ app.py                   # Código de la API REST
├─ modelo.pkl               # Modelo entrenado
├─ README.md                # Este archivo
├─ test_api.py              # Script para probar la API con ejemplos
└─ train_model.py           # Script para entrenar y guardar el modelo
└─ Dockerfile               # Instrucciones para construir la imagen Docker
```

---

## Ejecución Paso a Paso

- Los siguientes códigos deben ejecutarse en la terminal

### 1. Entrenar y guardar el modelo

```bash
python train_model.py
```

- Esto genera el archivo modelo.pkl.

---

![Salida entrenamiento](imagenes_readme/salida_1.png)

---

### 2. Construir imagen

```bash
docker build -t cancer-api .
```

- Esto genera la imagen cancer-api en base al Dockerfile que hayamos definido.

---

![Creación de imagen](imagenes_readme/salida_2.png)

---

### 3. Levantar la API

```bash
docker run -p 5000:5000 cancer-api
```

- Crea y ejecuta un contenedor a partir de la imagen **cancer-api**
- La API correrá en: `http://127.0.0.1:5000/`  

---

![API corriendo](imagenes_readme/salida_3.png)

---


### 4. Probar la API

```bash
curl.exe http://127.0.0.1:5000/
```
- Muestra un mensaje indicando que la API esta lista

---

![Mensaje](imagenes_readme/salida_4.png)

---

```bash
python test_api.py
```

- Muestra la predicción para 3 ejemplos:

---

![Predicciones](imagenes_readme/salida_5.png)

---

## Notas / Advertencias

- La API debe estar corriendo (ya sea con python app.py o dentro del contenedor con docker run …) para que test_api.py funcione.  
- Se recomienda usar un **entorno virtual** para evitar conflictos de librerías.  
- Revisar que el array enviado tenga exactamente **30 características**, como espera el modelo.
- Debido a lo anterior (30 características) el modelo no se desplegó en streamlit u otras plataformas de visualización y/o interacción.

---

## Autores

- Gonzalo Moyano Henríquez