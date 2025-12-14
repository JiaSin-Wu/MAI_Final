FROM python:3.10-slim

WORKDIR /weight-ensembling_MoE
COPY . .

# Install git (required for pip installing from git repositories)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt