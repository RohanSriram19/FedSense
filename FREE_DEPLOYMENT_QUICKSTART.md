# 🆓 One-Click FREE Deployment

Deploy your FedSense system for **absolutely free** in under 10 minutes!

## ⚡ FASTEST FREE DEPLOYMENT

### **Step 1: Frontend on Vercel (2 minutes)**

[![Deploy to Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/RohanSriram19/FedSense&project-name=fedsense-frontend&root-directory=frontend)

1. Click the button above
2. Connect your GitHub account
3. **Root Directory**: `frontend`
4. Click "Deploy"
5. Get your URL: `https://fedsense-frontend-xyz.vercel.app`

### **Step 2: Backend on Railway (3 minutes)**

1. **Go to**: https://railway.app/new
2. **Connect GitHub**: Select `RohanSriram19/FedSense`
3. **Choose**: "Deploy from GitHub repo"
4. **Dockerfile**: `Dockerfile.backend`
5. **Add Services**:
   - Click "+ New" 
   - Add "PostgreSQL" (free database)
6. **Deploy**: Click "Deploy"
7. Get your URL: `https://fedsense-backend-xyz.railway.app`

### **Step 3: Connect Frontend to Backend (2 minutes)**

1. **In Vercel**: Go to your project settings
2. **Environment Variables**: Add:
   ```
   NEXT_PUBLIC_API_URL=https://fedsense-backend-xyz.railway.app
   ```
3. **Redeploy**: Click "Deploy" to update

---

## 🎉 You're Live!

**Total Cost**: $0/month  
**Setup Time**: ~7 minutes  
**Features**: 
- ✅ Production-ready federated learning system
- ✅ Real-time ML dashboard
- ✅ HTTPS with custom domains
- ✅ Auto-deployments from GitHub
- ✅ Managed PostgreSQL database

---

## 🔄 Alternative Free Options

### **Option B: Render + Vercel**
- **Backend**: [Render.com](https://render.com) (Free tier)
- **Frontend**: [Vercel.com](https://vercel.com) (Free tier)
- **Database**: Render PostgreSQL (Free)

### **Option C: All-in-One Platforms**
- **Railway**: Full-stack deployment (500 hours/month free)
- **Fly.io**: 3 shared-cpu apps free
- **Koyeb**: Free tier with 512MB RAM

---

## 💡 Tips for Free Deployments

1. **Railway**: Best overall free option (500 hours = ~16 hours/day)
2. **Vercel**: Perfect for frontend (unlimited for personal use)  
3. **Render**: Good for backend but sleeps after 15min inactivity
4. **Supabase**: Free PostgreSQL with 500MB storage

---

## 🚀 Next Steps

1. **Deploy using Option 1** (Railway + Vercel)
2. **Get your public URLs**
3. **Share your federated learning system**
4. **Add it to your portfolio/resume**
5. **Upgrade later if needed**

Your production-quality FedSense system will be running on enterprise infrastructure for **$0**! 🎯
