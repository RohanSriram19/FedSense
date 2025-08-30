#!/bin/bash

# Railway Deployment Script for FedSense
echo "🚂 Deploying FedSense to Railway..."

# Check if railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Please install it first:"
    echo "npm install -g @railway/cli"
    exit 1
fi

# Check if user is logged in to Railway
if ! railway whoami &> /dev/null; then
    echo "🔐 Please login to Railway:"
    railway login
fi

# Create a new Railway project or use existing one
echo "📦 Setting up Railway project..."
railway link

# Set environment variables for Railway
echo "⚙️  Setting environment variables..."
railway variables set PYTHONPATH=/app
railway variables set MLFLOW_TRACKING_URI=sqlite:///app/mlflow.db
railway variables set NODE_ENV=production

# Deploy to Railway
echo "🚀 Deploying to Railway..."
railway up

echo "✅ Deployment initiated! Check Railway dashboard for progress."
echo "🌐 Your app will be available at the Railway-generated URL."

# Check deployment status
echo "📊 Checking deployment status..."
railway status

echo ""
echo "🎉 Railway deployment complete!"
echo "📱 Next step: Deploy frontend to Vercel with:"
echo "   cd frontend && npx vercel --prod"
