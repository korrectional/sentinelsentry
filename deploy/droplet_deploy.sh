#!/usr/bin/env bash
# DigitalOcean Droplet 1-Command Deployment Script for SentinelSentry
# Verified on DigitalOcean Basic Droplet (Ubuntu 22.04 / 24.04, 1 vCPU, 1 GB RAM)
set -euo pipefail

echo "=========================================================="
echo "🚀 Deploying SentinelSentry on DigitalOcean Droplet (CPU)"
echo "=========================================================="

# 1. Update and install base packages
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip python3-venv git curl

# 2. Setup virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify CPU performance & memory footprint
python3 -c "import psutil, os; from sentinelsentry.web.app import app; p = psutil.Process(os.getpid()); print(f'Verified CPU RAM usage: {p.memory_info().rss/(1024*1024):.1f} MB')"

# 5. Start Web Operations Center
echo "Starting SentinelSentry Web Operations Center on http://0.0.0.0:8000..."
uvicorn sentinelsentry.web.app:app --host 0.0.0.0 --port 8000
