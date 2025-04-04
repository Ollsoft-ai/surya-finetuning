FROM python:3.12.3-slim
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
   libgl1-mesa-glx \
   libglib2.0-0 \
   && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create necessary directories
RUN mkdir -p logs
# The container will wait for commands
CMD ["tail", "-f", "/dev/null"]