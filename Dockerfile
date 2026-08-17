# ══════════════════════════════════════════════════════════════════════════════
# CRAM Generator - Dockerfile
# ══════════════════════════════════════════════════════════════════════════════
# Multi-stage build for optimized production deployment
# Supports both AMD64 (x86_64) and ARM64 (Apple Silicon M1/M2/M3/M4)
# 
# Installation Order (mimics Colab restart behavior):
#   1. Install PyTorch first (isolated layer)
#   2. Install NumPy with compatible version (isolated layer)
#   3. Install DGL (Deep Graph Library)
#   4. Install remaining dependencies
#
# BUILD FOR M4/ARM64:
#   docker build --platform linux/arm64 -t cram-generator .
#
# BUILD FOR AMD64:
#   docker build --platform linux/amd64 -t cram-generator .
#
# RUN:
#   docker run -p 7860:7860 cram-generator
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: PyTorch Installation
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim AS pytorch-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    libxrender1 \
    libxext6 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch FIRST (CPU version for compatibility)
ARG TARGETPLATFORM
RUN pip install --upgrade pip && \
    pip install torch==2.2.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: NumPy Installation
# ─────────────────────────────────────────────────────────────────────────────
FROM pytorch-base AS numpy-base

# Install NumPy with compatible version
RUN pip install numpy==1.26.4

# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: DGL and Dependencies
# ─────────────────────────────────────────────────────────────────────────────
FROM numpy-base AS builder

# Install DGL (Deep Graph Library) for CPU
RUN pip install dgl -f https://data.dgl.ai/wheels/repo.html && \
    pip install dgllife

# Install remaining dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt || true

# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: Production Image
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim AS production

# Metadata
LABEL maintainer="CRAM Generator Team" \
      version="1.0.0" \
      description="CRAM Molecule Generator - Deep Learning for Marine DOM"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DGL_BACKEND=pytorch

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libxrender1 \
    libxext6 \
    libsm6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash cram
USER cram
WORKDIR /app

# Copy application code
COPY --chown=cram:cram . .

# Create output directories
RUN mkdir -p /app/outputs /app/data /app/models

# Expose Gradio port
EXPOSE 7860

# Health check using curl (installed in runtime deps)
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Default command - launch Gradio UI
CMD ["python", "ui/app.py", "--port", "7860"]

# ─────────────────────────────────────────────────────────────────────────────
# Alternative Commands:
# ─────────────────────────────────────────────────────────────────────────────
# Jupyter Notebook:
#   CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
#
# CLI Generate:
#   CMD ["python", "main.py", "generate", "--count", "1000"]
#
# Interactive Shell:
#   docker run -it cram-generator /bin/bash
# To run Jupyter instead:
# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
