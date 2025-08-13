import React, { useState, useEffect } from 'react';
import { Activity, CheckCircle, XCircle, AlertCircle, RefreshCw } from 'lucide-react';
import { AnalysisService } from '../services/AnalysisService';

interface BackendStatusProps {
  className?: string;
}

interface ModelStatus {
  [key: string]: boolean;
}

interface HealthStatus {
  status: string;
  timestamp?: string;
  uptime?: number;
  environment?: string;
  models?: ModelStatus;
}

export const BackendStatus: React.FC<BackendStatusProps> = ({ className = '' }) => {
  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null);
  const [modelStatus, setModelStatus] = useState<ModelStatus>({});
  const [isLoading, setIsLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const analysisService = AnalysisService.getInstance();

  const fetchStatus = async () => {
    setIsLoading(true);
    try {
      const [health, models] = await Promise.all([
        analysisService.getServiceHealth(),
        analysisService.getModelStatus()
      ]);
      
      setHealthStatus(health);
      setModelStatus(models);
      setLastUpdated(new Date());
    } catch (error) {
      console.error('Failed to fetch backend status:', error);
      setHealthStatus({ status: 'error' });
      setModelStatus({});
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    
    // Refresh status every 30 seconds
    const interval = setInterval(fetchStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'error':
        return <XCircle className="w-4 h-4 text-red-500" />;
      default:
        return <AlertCircle className="w-4 h-4 text-yellow-500" />;
    }
  };

  const getModelIcon = (isLoaded: boolean) => {
    return isLoaded ? 
      <CheckCircle className="w-4 h-4 text-green-500" /> : 
      <XCircle className="w-4 h-4 text-red-500" />;
  };

  const formatUptime = (seconds?: number) => {
    if (!seconds) return 'Unknown';
    
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    }
    return `${minutes}m`;
  };

  const formatTimestamp = (timestamp?: string) => {
    if (!timestamp) return 'Unknown';
    return new Date(timestamp).toLocaleTimeString();
  };

  return (
    <div className={`backend-status ${className}`}>
      <div className="status-header">
        <div className="flex items-center gap-2">
          <Activity className="w-5 h-5" />
          <h3 className="text-lg font-semibold">Backend Status</h3>
        </div>
        <button
          onClick={fetchStatus}
          disabled={isLoading}
          className="flex items-center gap-1 px-2 py-1 text-sm bg-blue-100 hover:bg-blue-200 rounded transition-colors disabled:opacity-50"
        >
          <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {isLoading ? (
        <div className="status-loading">
          <div className="animate-pulse">Loading status...</div>
        </div>
      ) : (
        <div className="status-content">
          {/* Health Status */}
          <div className="status-section">
            <h4 className="text-sm font-medium text-gray-700 mb-2">Service Health</h4>
            <div className="health-info">
              <div className="flex items-center gap-2">
                {getStatusIcon(healthStatus?.status || 'unknown')}
                <span className="capitalize">{healthStatus?.status || 'Unknown'}</span>
              </div>
              
              {healthStatus?.uptime && (
                <div className="text-xs text-gray-600">
                  Uptime: {formatUptime(healthStatus.uptime)}
                </div>
              )}
              
              {healthStatus?.timestamp && (
                <div className="text-xs text-gray-600">
                  Last Check: {formatTimestamp(healthStatus.timestamp)}
                </div>
              )}
              
              {healthStatus?.environment && (
                <div className="text-xs text-gray-600">
                  Environment: {healthStatus.environment}
                </div>
              )}
            </div>
          </div>

          {/* Model Status */}
          <div className="status-section">
            <h4 className="text-sm font-medium text-gray-700 mb-2">AI Models</h4>
            <div className="models-grid">
              {Object.entries(modelStatus).map(([model, isLoaded]) => (
                <div key={model} className="model-item">
                  <div className="flex items-center gap-2">
                    {getModelIcon(isLoaded)}
                    <span className="text-sm font-medium capitalize">{model}</span>
                  </div>
                  <span className="text-xs text-gray-600">
                    {isLoaded ? 'Loaded' : 'Not Available'}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Last Updated */}
          {lastUpdated && (
            <div className="text-xs text-gray-500 mt-2">
              Last updated: {lastUpdated.toLocaleTimeString()}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default BackendStatus;
