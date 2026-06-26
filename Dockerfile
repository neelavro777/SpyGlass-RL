FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./

# Install CPU-only PyTorch (much smaller than CUDA version)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code, data, and trained models
COPY app/ ./app/
COPY data/ ./data/
COPY notebooks/models/ ./notebooks/models/

EXPOSE 8501

WORKDIR /app/app

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
