# 🚀 FedSense Production Commands & Monitoring

## 📊 Service Management

### Start/Stop Services
```bash
# Start all services
docker-compose -f docker-compose.prod.yml up -d

# Stop all services  
docker-compose -f docker-compose.prod.yml down

# Restart specific service
docker-compose -f docker-compose.prod.yml restart backend

# View logs
docker-compose -f docker-compose.prod.yml logs -f backend
```

### Health Monitoring
```bash
# Check all services status
docker-compose -f docker-compose.prod.yml ps

# Check individual service health
curl http://localhost:8000/health          # Backend API
curl http://localhost:3000                 # Frontend
curl http://localhost:5000                 # MLflow
curl http://localhost:8001/v2/health       # Triton Inference
```

## 🔍 Monitoring & Debugging

### Resource Usage
```bash
# Monitor resource usage
docker stats

# Check specific container logs
docker logs fedsense-backend
docker logs fedsense-triton

# Execute commands inside containers
docker exec -it fedsense-backend bash
```

### Database Management
```bash
# Connect to PostgreSQL
docker exec -it fedsense-postgres psql -U fedsense_user -d fedsense

# Backup database
docker exec fedsense-postgres pg_dump -U fedsense_user fedsense > backup.sql
```

## 🎯 API Testing

### Backend API Endpoints
```bash
# Health check
curl http://localhost:8000/health

# List federated clients
curl http://localhost:8000/clients

# Get training metrics
curl http://localhost:8000/train/metrics

# Model information
curl http://localhost:8000/model/info

# Anomaly detection
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"data": [1.0, 2.0, 3.0, 4.0, 5.0]}'
```

### MLflow Integration
```bash
# MLflow experiments
curl http://localhost:5000/api/2.0/mlflow/experiments/list

# Model registry
curl http://localhost:5000/api/2.0/mlflow/registered-models/list
```

## 🔧 Development vs Production

### Switch Between Modes
```bash
# Development (hot reload)
make dev

# Production (optimized)
make prod
```

## 🛡️ Security & Maintenance

### SSL/HTTPS Setup (Production)
```bash
# Generate SSL certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx.key -out nginx.crt

# Update nginx.conf for HTTPS
# Restart nginx
docker-compose -f docker-compose.prod.yml restart nginx
```

### Backup & Recovery
```bash
# Full backup
make backup

# Restore from backup
make restore BACKUP_FILE=backup-2024-08-30.tar.gz
```

### Updates & Scaling
```bash
# Update to latest images
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d

# Scale services
docker-compose -f docker-compose.prod.yml up -d --scale backend=3
```

## 📈 Performance Monitoring

### Real-time Metrics
- **Frontend**: http://localhost:3000 (Dashboard with live charts)
- **Backend**: http://localhost:8000/docs (Swagger UI)
- **MLflow**: http://localhost:5000 (Experiment tracking)
- **Triton**: http://localhost:8001 (Model serving stats)

### Log Aggregation
```bash
# Follow all logs
docker-compose -f docker-compose.prod.yml logs -f

# Filter specific service
docker-compose -f docker-compose.prod.yml logs -f backend | grep ERROR
```

## 🎯 Production Checklist

- ✅ All services health checks passing
- ✅ Database connections established  
- ✅ Model artifacts loaded in Triton
- ✅ Frontend-backend API integration
- ✅ SSL certificates configured
- ✅ Monitoring and alerting setup
- ✅ Backup strategy implemented
- ✅ Load testing completed
