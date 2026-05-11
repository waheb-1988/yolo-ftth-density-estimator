# ── FALCON — Multi-stage Docker image ─────────────────────────────────────────
# Stage 1: dependencies
FROM python:3.11-slim AS base

WORKDIR /app

# System packages needed by OpenCV / PIL / ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 2: application ───────────────────────────────────────────────────────
FROM base AS app

WORKDIR /app
COPY . .

# Pre-download YOLOv8 nano weights so the container is self-contained
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || true

# Pre-generate demo data (safe to fail — regenerated on first launch)
RUN python scripts/build_demo.py --no-launch || true

# Expose both services
EXPOSE 8501 8000

# Default: run the Streamlit dashboard
CMD ["python", "run.py", "--demo", "--no-browser"]
