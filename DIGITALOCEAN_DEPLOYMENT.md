# 🌊 DigitalOcean Deployment Guide for FedSense

Deploy your federated learning system on DigitalOcean - simple, fast, and affordable!

## 🚀 Option 1: DigitalOcean App Platform (Easiest - 5 minutes)

**Best for**: Quick deployment, zero server management
**Cost**: ~$12-25/month | **Setup**: 5 minutes

### Step-by-step:
1. **Go to**: https://cloud.digitalocean.com/apps
2. **Create App**:
   - Source: GitHub
   - Repository: `RohanSriram19/FedSense`
   - Branch: `main`

3. **Configure Frontend**:
   - **Source Directory**: `frontend/`
   - **Build Command**: `npm ci && npm run build`
   - **Run Command**: `npm start`
   - **Environment**: `NODE_ENV=production`
   - **Port**: 3000

4. **Configure Backend**:
   - **Source Directory**: `/` (root)
   - **Dockerfile Path**: `Dockerfile.backend`
   - **Port**: 8000

5. **Add Database**: PostgreSQL (managed)
6. **Deploy**: Click "Create Resources"

**That's it!** DigitalOcean handles everything else.

---

## 🖥️ Option 2: DigitalOcean Droplet (More Control)

**Best for**: Full control, custom configuration
**Cost**: ~$24-48/month | **Setup**: 15 minutes

### Create Droplet:
- **Size**: 4GB RAM, 2 CPUs ($24/month)
- **Image**: Ubuntu 22.04 LTS
- **Region**: Choose closest to your users
- **SSH Keys**: Add your public key

### Auto-Deploy Script:
I'll create a one-command deployment script for you!

---

## 🐳 Option 3: DigitalOcean Kubernetes

**Best for**: Enterprise scale, high availability
**Cost**: ~$36-72/month | **Setup**: 20 minutes

Use your existing Kubernetes configs with DO managed Kubernetes.

---

## 💡 RECOMMENDED: App Platform

**Why DigitalOcean App Platform is perfect:**
- ✅ **$12/month** starting cost
- ✅ **Zero server maintenance**  
- ✅ **Auto-scaling**
- ✅ **Built-in SSL**
- ✅ **Custom domains**
- ✅ **GitHub integration**
- ✅ **Managed databases**

Much simpler than AWS!
