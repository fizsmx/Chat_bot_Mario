FROM python:3.12-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias para construir algunos paquetes (por si acaso)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar el archivo de dependencias
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código de la aplicación
COPY . .

# Exponer el puerto por el que funciona Streamlit
EXPOSE 8501

# Comando para ejecutar la aplicación
CMD ["chainlit", "run", "app.py", "--port", "8501", "--host", "0.0.0.0"]
