# FedSense Frontend Setup

The FedSense frontend is now running! 🎉

## Current Status

✅ **Frontend**: Running on http://localhost:3000  
✅ **Backend API**: Running on http://localhost:8000  
✅ **Full-Stack Integration**: API proxy configured  
✅ **Modern UI**: Next.js 14 + TypeScript + Tailwind CSS  
✅ **Real-time Dashboard**: Interactive anomaly detection visualization  

## Features

- **Dashboard Overview**: Real-time metrics and system status
- **Client Management**: Monitor federated learning participants  
- **Training Control**: Start/stop federated learning rounds
- **Anomaly Visualization**: Interactive charts with Recharts
- **Privacy Tracking**: Differential privacy budget monitoring
- **Model Analytics**: Performance metrics and evaluation results

## Architecture

```
Frontend (localhost:3000) -> API Proxy -> Backend (localhost:8000)
```

The Next.js app automatically proxies `/api/*` requests to the FastAPI backend, enabling seamless full-stack integration.

## Development Commands

```bash
# Frontend development
cd frontend/
npm run dev          # Start development server
npm run build        # Production build
npm run type-check   # TypeScript validation

# Backend development  
python -m fedsense.serve_fastapi    # Start API server
python -m fedsense.fl_server       # Start federated server
python -m fedsense.fl_client       # Start federated client
```

## Next Steps

1. **Explore Dashboard**: Open http://localhost:3000 to see the real-time dashboard
2. **Test API Integration**: The frontend automatically connects to the backend
3. **Run Federated Learning**: Use the dashboard controls to start training
4. **Monitor Privacy**: Track differential privacy budget usage
5. **Deploy**: Ready for production deployment with Docker

The FedSense system is now a complete full-stack application! 🚀
