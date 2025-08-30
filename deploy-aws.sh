#!/bin/bash

# FedSense AWS Deployment Script
# Run this script to deploy your FedSense system to AWS ECS + Fargate

set -e

# Configuration
REGION="us-east-1"
CLUSTER_NAME="fedsense-cluster"
FRONTEND_REPO="fedsense/frontend"
BACKEND_REPO="fedsense/backend"

echo "🚀 Starting FedSense AWS Deployment..."

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI not found. Install it first: https://aws.amazon.com/cli/"
    exit 1
fi

# Get AWS Account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "📋 Using AWS Account: $ACCOUNT_ID"

# Step 1: Create ECR Repositories
echo "📦 Creating ECR repositories..."
aws ecr create-repository --repository-name $FRONTEND_REPO --region $REGION || echo "Frontend repo already exists"
aws ecr create-repository --repository-name $BACKEND_REPO --region $REGION || echo "Backend repo already exists"

# Step 2: Login to ECR
echo "🔐 Logging into ECR..."
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

# Step 3: Build and Push Frontend
echo "🏗️  Building and pushing frontend..."
docker build -t fedsense-frontend -f frontend/Dockerfile .
docker tag fedsense-frontend:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$FRONTEND_REPO:latest
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$FRONTEND_REPO:latest

# Step 4: Build and Push Backend
echo "🏗️  Building and pushing backend..."
docker build -t fedsense-backend -f Dockerfile.backend .
docker tag fedsense-backend:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$BACKEND_REPO:latest
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$BACKEND_REPO:latest

# Step 5: Update task definitions with account ID
echo "📝 Updating task definitions..."
sed -i '' "s/ACCOUNT_ID/$ACCOUNT_ID/g" aws-ecs-frontend-task.json
sed -i '' "s/ACCOUNT_ID/$ACCOUNT_ID/g" aws-ecs-backend-task.json

# Step 6: Create ECS Cluster
echo "🎯 Creating ECS cluster..."
aws ecs create-cluster --cluster-name $CLUSTER_NAME --capacity-providers FARGATE --region $REGION || echo "Cluster already exists"

# Step 7: Create CloudWatch Log Groups
echo "📊 Creating CloudWatch log groups..."
aws logs create-log-group --log-group-name /ecs/fedsense-frontend --region $REGION || echo "Frontend log group exists"
aws logs create-log-group --log-group-name /ecs/fedsense-backend --region $REGION || echo "Backend log group exists"

# Step 8: Register Task Definitions
echo "📋 Registering ECS task definitions..."
aws ecs register-task-definition --cli-input-json file://aws-ecs-frontend-task.json --region $REGION
aws ecs register-task-definition --cli-input-json file://aws-ecs-backend-task.json --region $REGION

# Step 9: Create services (requires VPC and subnets)
echo "⚠️  Manual step required:"
echo "1. Go to AWS ECS Console"
echo "2. Create services using the registered task definitions"
echo "3. Configure Application Load Balancer"
echo "4. Set up custom domain with Route 53"

echo "✅ Docker images pushed successfully!"
echo "🎯 ECR Frontend: $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$FRONTEND_REPO:latest"
echo "🎯 ECR Backend: $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$BACKEND_REPO:latest"
echo ""
echo "🌐 Next steps:"
echo "1. Create RDS PostgreSQL database"
echo "2. Set up Application Load Balancer"
echo "3. Create ECS services in AWS Console"
echo "4. Configure domain name with Route 53"
echo ""
echo "💡 Estimated monthly cost: $50-100"
