FROM python:3.11.15-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements-lock.txt ./
RUN python -m pip install --no-cache-dir -r requirements-lock.txt

COPY . ./

EXPOSE 8501

# Data is installed at container runtime into the mounted .data volume. This
# keeps the frozen checkpoint out of both Git history and the image layers.
CMD ["sh", "-c", "python scripts/bootstrap_dashboard_data.py && exec streamlit run apps/inferencex_pca_demo.py --server.address=0.0.0.0 --server.port=8501"]
