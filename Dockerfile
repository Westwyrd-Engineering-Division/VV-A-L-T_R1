# V.V.A.L.T Production Dockerfile
FROM python:3.9-slim

LABEL maintainer="V.V.A.L.T Contributors"
LABEL description="Vantage-Vector Autonomous Logic Transformer"
LABEL version="0.1.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY vvalt/ vvalt/
COPY setup.py .
COPY README.md .
COPY LICENSE .

# Install package
RUN pip install --no-cache-dir .

# Create non-root user
RUN useradd -m -u 1000 vvalt && \
    chown -R vvalt:vvalt /app

USER vvalt

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "from vvalt import VVALT; VVALT(10, 8, 5)" || exit 1

# Default command
CMD ["python"]

# Metadata
EXPOSE 8080
ENV VVALT_LOG_LEVEL=INFO
ENV PYTHONUNBUFFERED=1
