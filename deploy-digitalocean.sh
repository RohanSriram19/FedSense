#!/bin/bash

# FedSense DigitalOcean Droplet Deployment Script
# One-command deployment to DigitalOcean Ubuntu droplet

set -e

echo "🌊 FedSense DigitalOcean Deployment Starting..."

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
echo "🐳 Installing Docker..."
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
echo "📦 Installing Docker Compose..."
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install Git
sudo apt install git -y

# Clone repository
echo "📂 Cloning FedSense repository..."
if [ -d "FedSense" ]; then
    cd FedSense
    git pull
else
    git clone https://github.com/RohanSriram19/FedSense.git
    cd FedSense
fi

# Create environment file
echo "⚙️  Creating environment configuration..."
cat > .env.prod << EOF
# Production Environment Variables
DATABASE_URL=postgresql://postgres:fedsense_secure_2024@postgres:5432/fedsense
MLFLOW_BACKEND_STORE_URI=postgresql://mlflow:mlflow_secure_2024@postgres:5432/mlflow
REDIS_URL=redis://redis:6379/0
ENV=production
DEBUG=false

# Security
SECRET_KEY=$(openssl rand -hex 32)
JWT_SECRET=$(openssl rand -hex 32)

# External URLs (update after getting domain)
FRONTEND_URL=http://YOUR_DOMAIN_OR_IP
BACKEND_URL=http://YOUR_DOMAIN_OR_IP/api
EOF

# Start services
echo "🚀 Starting FedSense services..."
docker-compose -f docker-compose.prod.yml up -d

# Setup firewall
echo "🛡️  Configuring firewall..."
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw --force enable

# Install and configure Nginx
echo "🌐 Setting up Nginx reverse proxy..."
sudo apt install nginx -y

# Create Nginx configuration
sudo cat > /etc/nginx/sites-available/fedsense << EOF
server {
    listen 80;
    server_name _;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";

    # Frontend
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_cache_bypass \$http_upgrade;
    }

    # Backend API
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # MLflow
    location /mlflow/ {
        proxy_pass http://localhost:5001/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# Enable site
sudo ln -sf /etc/nginx/sites-available/fedsense /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx

# Install SSL certificate tool
sudo apt install snapd -y
sudo snap install --classic certbot
sudo ln -s /snap/bin/certbot /usr/bin/certbot

# Get server IP
SERVER_IP=$(curl -s ifconfig.me)

echo "✅ FedSense deployment completed!"
echo ""
echo "🌐 Your FedSense system is now live at:"
echo "   http://$SERVER_IP"
echo ""
echo "📋 Service Status:"
docker-compose -f docker-compose.prod.yml ps

echo ""
echo "🔒 To enable HTTPS with custom domain:"
echo "1. Point your domain to: $SERVER_IP"
echo "2. Run: sudo certbot --nginx -d yourdomain.com"
echo ""
echo "📊 Monitor logs:"
echo "   docker-compose -f docker-compose.prod.yml logs -f"
echo ""
echo "💰 Estimated DigitalOcean cost: \$24-48/month"
echo "🎉 Happy federated learning!"
