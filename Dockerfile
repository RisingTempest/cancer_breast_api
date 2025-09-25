# Imagen base de Python
FROM python:3.10-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar primero requirements.txt y instalar (cache más eficiente)
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de los archivos
COPY modelo.pkl .
COPY app.py .

# Exponer el puerto donde correrá Flask
EXPOSE 5001

# Comando por defecto al correr el contenedor
CMD ["python", "app.py"]
