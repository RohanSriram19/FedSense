'use client'

import { useEffect, useState } from 'react'
import { apiClient, type ClientStatus, type TrainingMetrics, type ModelInfo, type AnomalyResult } from '@/lib/api'
import { cn, formatNumber, formatPercentage, formatDuration } from '@/lib/utils'
import { 
  Activity, 
  Users, 
  TrendingUp, 
  Shield, 
  Play, 
  Pause, 
  AlertTriangle,
  CheckCircle,
  Clock,
  Database,
  Brain,
  Settings
} from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, BarChart, Bar } from 'recharts'

interface DashboardStats {
  totalClients: number
  activeClients: number
  trainingRounds: number
  modelAccuracy: number
  anomaliesDetected: number
  privacyBudgetUsed: number
}

export default function Dashboard() {
  const [stats, setStats] = useState<DashboardStats>({
    totalClients: 0,
    activeClients: 0,
    trainingRounds: 0,
    modelAccuracy: 0,
    anomaliesDetected: 0,
    privacyBudgetUsed: 0
  })
  const [clients, setClients] = useState<ClientStatus[]>([])
  const [trainingMetrics, setTrainingMetrics] = useState<TrainingMetrics[]>([])
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null)
  const [recentAnomalies, setRecentAnomalies] = useState<AnomalyResult[]>([])
  const [isTraining, setIsTraining] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchData = async () => {
    try {
      setError(null)
      const [clientsData, metricsData, modelData] = await Promise.all([
        apiClient.getClients(),
        apiClient.getTrainingMetrics(),
        apiClient.getModelInfo()
      ])

      setClients(clientsData)
      setTrainingMetrics(metricsData)
      setModelInfo(modelData)

      // Calculate dashboard stats
      const activeClients = clientsData.filter(c => c.status === 'online' || c.status === 'training').length
      const latestMetrics = metricsData[metricsData.length - 1]
      
      setStats({
        totalClients: clientsData.length,
        activeClients,
        trainingRounds: metricsData.length,
        modelAccuracy: latestMetrics?.accuracy || modelData.accuracy,
        anomaliesDetected: Math.floor(Math.random() * 50), // Mock data
        privacyBudgetUsed: 0.65 // Mock data
      })

      // Mock recent anomalies data
      setRecentAnomalies([
        { timestamp: '2024-01-20T10:30:00Z', value: 0.85, anomaly_score: 0.92, is_anomaly: true, confidence: 0.88 },
        { timestamp: '2024-01-20T09:45:00Z', value: 0.23, anomaly_score: 0.15, is_anomaly: false, confidence: 0.95 },
        { timestamp: '2024-01-20T09:15:00Z', value: 0.78, anomaly_score: 0.87, is_anomaly: true, confidence: 0.82 },
      ])

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch data')
    } finally {
      setLoading(false)
    }
  }

  const handleTrainingToggle = async () => {
    try {
      if (isTraining) {
        await apiClient.stopTraining()
        setIsTraining(false)
      } else {
        await apiClient.startTraining()
        setIsTraining(true)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to toggle training')
    }
  }

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 30000) // Refresh every 30 seconds
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary"></div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <AlertTriangle className="h-16 w-16 text-red-500 mx-auto mb-4" />
          <h2 className="text-2xl font-bold mb-2">Error Loading Dashboard</h2>
          <p className="text-muted-foreground mb-4">{error}</p>
          <button 
            onClick={fetchData}
            className="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 space-y-4 p-8 pt-6">
      <div className="flex items-center justify-between space-y-2">
        <h2 className="text-3xl font-bold tracking-tight">FedSense Dashboard</h2>
        <div className="flex items-center space-x-2">
          <button
            onClick={handleTrainingToggle}
            className={cn(
              "flex items-center gap-2 px-4 py-2 rounded-md font-medium transition-colors",
              isTraining 
                ? "bg-red-500 hover:bg-red-600 text-white"
                : "bg-green-500 hover:bg-green-600 text-white"
            )}
          >
            {isTraining ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            {isTraining ? 'Stop Training' : 'Start Training'}
          </button>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <div className="metric-card">
          <div className="flex flex-row items-center justify-between space-y-0 pb-2">
            <div className="text-sm font-medium">Total Clients</div>
            <Users className="h-4 w-4 text-muted-foreground" />
          </div>
          <div className="text-2xl font-bold">{stats.totalClients}</div>
          <p className="text-xs text-muted-foreground">
            {stats.activeClients} active
          </p>
        </div>
        
        <div className="metric-card">
          <div className="flex flex-row items-center justify-between space-y-0 pb-2">
            <div className="text-sm font-medium">Model Accuracy</div>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </div>
          <div className="text-2xl font-bold">{formatPercentage(stats.modelAccuracy)}</div>
          <p className="text-xs text-muted-foreground">
            +2.1% from last round
          </p>
        </div>

        <div className="metric-card">
          <div className="flex flex-row items-center justify-between space-y-0 pb-2">
            <div className="text-sm font-medium">Training Rounds</div>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </div>
          <div className="text-2xl font-bold">{stats.trainingRounds}</div>
          <p className="text-xs text-muted-foreground">
            {isTraining ? 'Training in progress' : 'Last completed'}
          </p>
        </div>

        <div className="metric-card">
          <div className="flex flex-row items-center justify-between space-y-0 pb-2">
            <div className="text-sm font-medium">Privacy Budget</div>
            <Shield className="h-4 w-4 text-muted-foreground" />
          </div>
          <div className="text-2xl font-bold">{formatPercentage(stats.privacyBudgetUsed)}</div>
          <p className="text-xs text-muted-foreground">
            {formatPercentage(1 - stats.privacyBudgetUsed)} remaining
          </p>
        </div>
      </div>

      {/* Charts Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
        <div className="col-span-4">
          <div className="metric-card">
            <div className="flex flex-col space-y-1.5 p-6">
              <div className="text-base font-semibold">Training Progress</div>
              <div className="text-sm text-muted-foreground">Model accuracy over training rounds</div>
            </div>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={trainingMetrics}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="round" />
                  <YAxis domain={[0, 1]} />
                  <Tooltip 
                    formatter={(value: number) => [formatPercentage(value), 'Accuracy']}
                    labelFormatter={(round: number) => `Round ${round}`}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="accuracy" 
                    stroke="#8884d8" 
                    strokeWidth={2}
                    dot={{ fill: '#8884d8', strokeWidth: 2, r: 4 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        <div className="col-span-3">
          <div className="metric-card">
            <div className="flex flex-col space-y-1.5 p-6">
              <div className="text-base font-semibold">Client Status</div>
              <div className="text-sm text-muted-foreground">Real-time client connectivity</div>
            </div>
            <div className="px-6 pb-6">
              <div className="space-y-3">
                {clients.slice(0, 6).map((client) => (
                  <div key={client.client_id} className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <div className={cn(
                        "h-2 w-2 rounded-full",
                        client.status === 'online' && 'bg-green-500',
                        client.status === 'offline' && 'bg-red-500',
                        client.status === 'training' && 'bg-blue-500'
                      )} />
                      <span className="text-sm font-medium">{client.client_id}</span>
                    </div>
                    <div className="flex items-center space-x-2 text-xs text-muted-foreground">
                      <Database className="h-3 w-3" />
                      <span>{client.data_samples}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Anomalies */}
      <div className="metric-card">
        <div className="flex flex-col space-y-1.5 p-6 pb-4">
          <div className="text-base font-semibold">Recent Anomalies</div>
          <div className="text-sm text-muted-foreground">Latest anomaly detections from all clients</div>
        </div>
        <div className="px-6 pb-6">
          <div className="space-y-3">
            {recentAnomalies.map((anomaly, index) => (
              <div key={index} className={cn(
                "flex items-center justify-between p-3 rounded-lg border",
                anomaly.is_anomaly ? "anomaly-alert" : "bg-muted/50"
              )}>
                <div className="flex items-center space-x-3">
                  {anomaly.is_anomaly ? (
                    <AlertTriangle className="h-4 w-4 text-orange-600" />
                  ) : (
                    <CheckCircle className="h-4 w-4 text-green-600" />
                  )}
                  <div>
                    <div className="text-sm font-medium">
                      {anomaly.is_anomaly ? 'Anomaly Detected' : 'Normal Reading'}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      Score: {formatNumber(anomaly.anomaly_score)} | 
                      Confidence: {formatPercentage(anomaly.confidence)}
                    </div>
                  </div>
                </div>
                <div className="text-right text-xs text-muted-foreground">
                  <Clock className="h-3 w-3 inline mr-1" />
                  {new Date(anomaly.timestamp).toLocaleTimeString()}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
