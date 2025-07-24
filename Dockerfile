# Use official Python image
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev \
        libgl1-mesa-glx \
        && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the code
COPY . .

# Set environment variables (optional)
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "run_rl_pipeline.py"] 