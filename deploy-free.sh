#!/bin/bash

# Free Deployment Script for FedSense
# Deploy to Railway (Backend) + Vercel (Frontend) for $0

echo "🆓 Starting FREE FedSense Deployment..."

# Check dependencies
echo "🔍 Checking requirements..."

if ! command -v gh &> /dev/null; then
    echo "📦 Installing GitHub CLI..."
    brew install gh || (echo "❌ Please install GitHub CLI: https://cli.github.com/" && exit 1)
fi

echo "✅ Prerequisites ready!"

echo ""
echo "🚀 FREE DEPLOYMENT OPTIONS:"
echo ""
echo "1️⃣  RAILWAY + VERCEL (Recommended)"
echo "   Frontend: Vercel (Free forever)"
echo "   Backend: Railway (500 hours/month free)"
echo "   Database: Railway PostgreSQL (Free)"
echo "   Cost: $0/month"
echo ""
echo "2️⃣  RENDER + VERCEL"  
echo "   Frontend: Vercel (Free forever)"
echo "   Backend: Render (Free with sleep)"
echo "   Database: Supabase (Free PostgreSQL)"
echo "   Cost: $0/month"
echo ""
echo "3️⃣  ALL VERCEL (Hobby Plan)"
echo "   Full-stack: Vercel (Free tier)"
echo "   Database: Planetscale (Free tier)"
echo "   Cost: $0/month"
echo ""

read -p "Choose option (1-3): " choice

case $choice in
    1)
        echo "🚂 Setting up Railway + Vercel deployment..."
        echo ""
        echo "📋 RAILWAY BACKEND DEPLOYMENT:"
        echo "1. Go to: https://railway.app/new"
        echo "2. Connect GitHub: RohanSriram19/FedSense"
        echo "3. Select 'Deploy from GitHub repo'"
        echo "4. Choose Dockerfile: Dockerfile.backend"
        echo "5. Add PostgreSQL database (free)"
        echo "6. Deploy!"
        echo ""
        echo "🔗 VERCEL FRONTEND DEPLOYMENT:"
        echo "1. Go to: https://vercel.com/new"
        echo "2. Import: RohanSriram19/FedSense"
        echo "3. Root Directory: frontend/"
        echo "4. Framework: Next.js"
        echo "5. Deploy!"
        ;;
    2)
        echo "🎨 Setting up Render + Vercel deployment..."
        echo ""
        echo "📋 RENDER BACKEND DEPLOYMENT:"
        echo "1. Go to: https://render.com/new/web"
        echo "2. Connect GitHub: RohanSriram19/FedSense"  
        echo "3. Use Dockerfile: Dockerfile.backend"
        echo "4. Add PostgreSQL database (free)"
        echo "5. Deploy!"
        echo ""
        echo "🔗 VERCEL FRONTEND DEPLOYMENT:"
        echo "1. Go to: https://vercel.com/new"
        echo "2. Import: RohanSriram19/FedSense"
        echo "3. Root Directory: frontend/"
        echo "4. Deploy!"
        ;;
    3)
        echo "⚡ Setting up All-Vercel deployment..."
        echo ""
        echo "🔗 VERCEL FULL-STACK DEPLOYMENT:"
        echo "1. Go to: https://vercel.com/new"
        echo "2. Import: RohanSriram19/FedSense"
        echo "3. Framework: Next.js"
        echo "4. Add Planetscale database"
        echo "5. Deploy!"
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "🎉 Deployment instructions ready!"
echo ""
echo "💡 After deployment, you'll get:"
echo "   ✅ Public HTTPS URLs"
echo "   ✅ Auto-deployments from GitHub"
echo "   ✅ SSL certificates"
echo "   ✅ Custom domains (optional)"
echo ""
echo "📊 Your federated learning system will be live for FREE!"
echo "🌍 Share your URLs and showcase your ML expertise!"
