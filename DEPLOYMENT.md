# FedSense Deployment Guide 🚀

This guide covers multiple deployment options for the FedSense federated learning platform.

## 🐳 Docker Deployment (Recommended)

### Prerequisites
- Docker and Docker Compose installed
- At least 4GB RAM and 2 CPU cores
- 10GB available disk space

### Quick Start
```bash
# Clone the repository
git clone https://github.com/RohanSriram19/FedSense.git
cd FedSense

# Start all services
docker-compose -f docker-compose.prod.yml up -d

# Check service status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f
```

### Services
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **MLflow**: http://localhost:5000
- **API Documentation**: http://localhost:8000/docs

### Environment Variables
Create a `.env` file:
```bash
# Database
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=mlflow

# MLflow
MLFLOW_TRACKING_URI=postgresql://mlflow:password@postgres:5432/mlflow

# API Configuration
FEDSENSE_LOG_LEVEL=INFO
FEDSENSE_CORS_ORIGINS=["http://localhost:3000"]
```

## ☸️ Kubernetes Deployment

### Prerequisites
- Kubernetes cluster (minikube, EKS, GKE, AKS)
- kubectl configured
- Helm (optional, for package management)

### Deploy to Kubernetes
```bash
# Apply all configurations
kubectl apply -f k8s/

# Check deployment status
kubectl get pods
kubectl get services

# Get external IP (for LoadBalancer)
kubectl get service fedsense-frontend-service

# Port forward for local access
kubectl port-forward service/fedsense-frontend-service 3000:3000
kubectl port-forward service/fedsense-backend-service 8000:8000
```

### Scaling
```bash
# Scale backend replicas
kubectl scale deployment fedsense-backend --replicas=5

# Scale frontend replicas
kubectl scale deployment fedsense-frontend --replicas=3

# Check horizontal pod autoscaler
kubectl get hpa
```

## ☁️ Cloud Platform Deployment

### AWS Deployment

#### Option 1: AWS ECS with Fargate
```bash
# Install AWS CLI and configure credentials
aws configure

# Create ECS cluster
aws ecs create-cluster --cluster-name fedsense-cluster

# Build and push images to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com

# Tag and push images
docker tag fedsense/backend:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/fedsense-backend:latest
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/fedsense-backend:latest

# Deploy using ECS task definitions (see aws/ directory)
```

#### Option 2: AWS App Runner
```bash
# Create apprunner.yaml in root directory
version: 1.0
runtime: docker
build:
  commands:
    build:
      - echo "Build started on `date`"
      - docker build -t fedsense-backend -f Dockerfile.backend .
run:
  runtime-version: latest
  command: python -m fedsense.serve_fastapi --host 0.0.0.0 --port 8000
  network:
    port: 8000
    env: PORT
```

### Google Cloud Platform

#### Cloud Run Deployment
```bash
# Install gcloud CLI and authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Build and deploy backend
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/fedsense-backend
gcloud run deploy fedsense-backend \
  --image gcr.io/YOUR_PROJECT_ID/fedsense-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated

# Build and deploy frontend
cd frontend
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/fedsense-frontend
gcloud run deploy fedsense-frontend \
  --image gcr.io/YOUR_PROJECT_ID/fedsense-frontend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### GKE Deployment
```bash
# Create GKE cluster
gcloud container clusters create fedsense-cluster \
  --num-nodes=3 \
  --machine-type=e2-standard-2 \
  --zone=us-central1-a

# Get credentials
gcloud container clusters get-credentials fedsense-cluster --zone=us-central1-a

# Deploy using kubectl
kubectl apply -f k8s/
```

### Azure Deployment

#### Azure Container Instances
```bash
# Install Azure CLI and login
az login

# Create resource group
az group create --name fedsense-rg --location eastus

# Deploy backend container
az container create \
  --resource-group fedsense-rg \
  --name fedsense-backend \
  --image ghcr.io/rohansriram19/fedsense/backend:latest \
  --dns-name-label fedsense-backend-unique \
  --ports 8000

# Deploy frontend container
az container create \
  --resource-group fedsense-rg \
  --name fedsense-frontend \
  --image ghcr.io/rohansriram19/fedsense/frontend:latest \
  --dns-name-label fedsense-frontend-unique \
  --ports 3000 \
  --environment-variables NEXT_PUBLIC_API_URL=http://fedsense-backend-unique.eastus.azurecontainer.io:8000
```

## 🔧 Configuration

### Environment Variables

#### Backend (.env)
```bash
# Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=4

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/fedsense

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# Triton Inference Server
TRITON_URL=http://localhost:8001

# Federated Learning
FL_ROUNDS=10
FL_MIN_CLIENTS=2
FL_SAMPLE_FRACTION=1.0

# Privacy
DP_EPSILON=1.0
DP_DELTA=1e-5
DP_NOISE_MULTIPLIER=0.1

# Security
SECRET_KEY=your-secret-key-here
CORS_ORIGINS=["http://localhost:3000"]

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

#### Frontend (.env.local)
```bash
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000

# Environment
NODE_ENV=production
NEXT_TELEMETRY_DISABLED=1

# Analytics (optional)
NEXT_PUBLIC_GOOGLE_ANALYTICS=GA_MEASUREMENT_ID
```

## 📊 Monitoring and Observability

### Health Checks
```bash
# Backend health
curl http://localhost:8000/health

# Frontend health
curl http://localhost:3000/api/health

# MLflow health
curl http://localhost:5000/health
```

### Metrics and Logging
- **Application Metrics**: Available at `/metrics` endpoint (Prometheus format)
- **MLflow Tracking**: Model experiments and metrics
- **Container Logs**: Structured JSON logging
- **Health Checks**: Built-in health check endpoints

### Monitoring Stack (Optional)
```yaml
# Add to docker-compose.prod.yml
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

## 🔐 Security Considerations

### Production Security Checklist
- [ ] Use HTTPS/TLS certificates
- [ ] Set strong database passwords
- [ ] Configure firewall rules
- [ ] Enable authentication for MLflow
- [ ] Use secrets management (AWS Secrets Manager, etc.)
- [ ] Enable container scanning
- [ ] Set up monitoring and alerting
- [ ] Configure backup strategies
- [ ] Implement rate limiting
- [ ] Use non-root containers

### SSL/TLS Configuration
```bash
# Generate self-signed certificates (for testing)
mkdir -p docker/nginx/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout docker/nginx/ssl/key.pem \
  -out docker/nginx/ssl/cert.pem
```

## 🚨 Troubleshooting

### Common Issues

#### Container Won't Start
```bash
# Check logs
docker-compose logs fedsense-api
docker-compose logs fedsense-frontend

# Check resource usage
docker stats

# Restart services
docker-compose restart fedsense-api
```

#### API Connection Issues
```bash
# Test network connectivity
docker-compose exec fedsense-frontend curl http://fedsense-api:8000/health

# Check environment variables
docker-compose exec fedsense-frontend env | grep API_URL
```

#### Database Connection Issues
```bash
# Check PostgreSQL status
docker-compose exec postgres psql -U mlflow -d mlflow -c "SELECT version();"

# Reset database
docker-compose down -v
docker-compose up postgres -d
```

### Performance Tuning

#### Backend Optimization
```bash
# Increase worker processes
export WORKERS=8

# Tune memory limits
docker-compose up --scale fedsense-api=3
```

#### Frontend Optimization
```bash
# Enable Next.js optimization
export NEXT_BUILD_TARGET=serverless

# Configure caching
export NEXT_CACHE_TIMEOUT=3600
```

## 📞 Support

For deployment issues:
1. Check the [troubleshooting guide](#troubleshooting)
2. Review application logs
3. Open an issue on GitHub
4. Check the documentation at `/docs` endpoint

---

**Happy Deploying! 🚀**
