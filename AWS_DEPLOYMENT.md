# 🚀 AWS Deployment Guide for FedSense

Your federated learning system can be deployed on AWS in multiple ways. Here are the best options from quickest to most production-ready.

## ☁️ Option 1: AWS App Runner (Easiest - 5 minutes)

**Best for**: Quick deployment, automatic scaling, minimal setup

### Deploy Frontend to App Runner:
```bash
# 1. Push your code to GitHub (already done)
# 2. Go to AWS App Runner console
# 3. Create service from GitHub
# 4. Point to: https://github.com/RohanSriram19/FedSense
# 5. Use: frontend/Dockerfile
```

### Deploy Backend to App Runner:
```bash
# Create second App Runner service
# Use: Dockerfile.backend
# Environment variables for database connections
```

**Cost**: ~$25-50/month for both services

---

## ☁️ Option 2: AWS ECS + Fargate (Recommended)

**Best for**: Production-ready, container orchestration, cost-effective

I'll create the deployment files for you:

### Step 1: Build and Push to ECR
```bash
# Create ECR repositories
aws ecr create-repository --repository-name fedsense/frontend
aws ecr create-repository --repository-name fedsense/backend

# Get login token
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Build and push
docker build -t fedsense-frontend -f frontend/Dockerfile .
docker tag fedsense-frontend:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/fedsense/frontend:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/fedsense/frontend:latest

docker build -t fedsense-backend -f Dockerfile.backend .
docker tag fedsense-backend:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/fedsense/backend:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/fedsense/backend:latest
```

**Cost**: ~$50-100/month

---

## ☁️ Option 3: AWS EC2 + Docker (Most Control)

**Best for**: Full control, custom configuration, cost optimization

### Launch EC2 Instance:
```bash
# Use AWS CLI or Console
# Instance type: t3.large (8GB RAM for ML workloads)
# AMI: Amazon Linux 2
# Security Groups: 80, 443, 8000, 3000
```

### Deploy Script:
```bash
#!/bin/bash
# Auto-deployment script for EC2

# Install Docker
sudo yum update -y
sudo amazon-linux-extras install docker -y
sudo service docker start
sudo usermod -a -G docker ec2-user

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Clone and deploy
git clone https://github.com/RohanSriram19/FedSense.git
cd FedSense
docker-compose -f docker-compose.prod.yml up -d

# Setup reverse proxy with SSL
sudo yum install nginx certbot python3-certbot-nginx -y
sudo systemctl start nginx
sudo certbot --nginx -d yourdomain.com
```

**Cost**: ~$30-60/month

---

## ☁️ Option 4: AWS EKS (Enterprise Scale)

**Best for**: High availability, auto-scaling, enterprise features

Use your existing Kubernetes configs:
```bash
# Create EKS cluster
eksctl create cluster --name fedsense-cluster --region us-east-1

# Deploy
kubectl apply -f k8s/
```

**Cost**: ~$150-300/month

---

## 🎯 FASTEST AWS DEPLOYMENT (Let's do this!)

Let me set up **Option 2 (ECS + Fargate)** for you:
