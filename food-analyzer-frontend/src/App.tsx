import React, { useState } from 'react';
import { Camera, BarChart3, History, TrendingUp, Settings } from 'lucide-react';
import ImageUpload from './components/ImageUpload';
import AnalysisResults from './components/AnalysisResults';
import NutritionCharts from './components/NutritionCharts';
import HistoryView from './components/HistoryView';
import TrendsView from './components/TrendsView';
import BackendStatus from './components/BackendStatus';
import type { AnalysisResult, HistoryEntry } from './types';
import { HistoryService } from './services/HistoryService';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState<'upload' | 'results' | 'charts' | 'history' | 'trends' | 'status'>('upload');
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleAnalysisComplete = (result: AnalysisResult, imageFile?: File) => {
    setAnalysisResult(result);

    // Add to history
    const historyEntry: HistoryEntry = {
      id: Date.now().toString(),
      timestamp: new Date(),
      analysis_result: result,
      image_url: imageFile ? URL.createObjectURL(imageFile) : undefined
    };

    // Save to localStorage and update state
    const historyService = HistoryService.getInstance();
    historyService.saveToHistory(historyEntry);
    setHistory(prev => [historyEntry, ...prev]);
    setActiveTab('results');
  };

  // Load history on component mount
  React.useEffect(() => {
    const historyService = HistoryService.getInstance();
    const savedHistory = historyService.getHistory();
    setHistory(savedHistory);
  }, []);

  const tabs = [
    { id: 'upload' as const, label: 'Analyze Food', icon: Camera },
    { id: 'results' as const, label: 'Results', icon: BarChart3 },
    { id: 'charts' as const, label: 'Nutrition Charts', icon: BarChart3 },
    { id: 'history' as const, label: 'History', icon: History },
    { id: 'trends' as const, label: 'Trends', icon: TrendingUp },
    { id: 'status' as const, label: 'System Status', icon: Settings }
  ];

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <div className="logo">
            <Camera className="logo-icon" />
            <h1>AI Calorie Analyzer</h1>
          </div>
          <p className="tagline">Advanced food recognition and nutritional analysis</p>
        </div>
      </header>

      <nav className="tab-navigation">
        {tabs.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            className={`tab-button ${activeTab === id ? 'active' : ''}`}
            onClick={() => setActiveTab(id)}
            disabled={id === 'results' && !analysisResult}
          >
            <Icon size={20} />
            <span>{label}</span>
          </button>
        ))}
      </nav>

      <main className="main-content">
        {activeTab === 'upload' && (
          <div className="tab-content">
            <ImageUpload
              onAnalysisComplete={handleAnalysisComplete}
              isAnalyzing={isAnalyzing}
              setIsAnalyzing={setIsAnalyzing}
            />
          </div>
        )}

        {activeTab === 'results' && analysisResult && (
          <div className="tab-content">
            <AnalysisResults result={analysisResult} />
          </div>
        )}

        {activeTab === 'charts' && analysisResult && (
          <div className="tab-content">
            <NutritionCharts data={analysisResult.nutritional_data} />
          </div>
        )}

        {activeTab === 'history' && (
          <div className="tab-content">
            <HistoryView
              history={history}
              onSelectEntry={(entry) => {
                setAnalysisResult(entry.analysis_result);
                setActiveTab('results');
              }}
            />
          </div>
        )}

        {activeTab === 'trends' && (
          <div className="tab-content">
            <TrendsView />
          </div>
        )}

        {activeTab === 'status' && (
          <div className="tab-content">
            <div className="status-container">
              <BackendStatus />
              <div className="status-info">
                <h3>System Information</h3>
                <p>This application uses a hybrid TypeScript backend with Python AI models for advanced food recognition and nutritional analysis.</p>
                <div className="features-list">
                  <h4>Key Features:</h4>
                  <ul>
                    <li>Multiple AI models (YOLO, ViT, Swin, BLIP, CLIP, LLM)</li>
                    <li>Real-time food detection and recognition</li>
                    <li>Comprehensive nutritional analysis</li>
                    <li>Advanced ensemble detection</li>
                    <li>Health monitoring and status tracking</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      <footer className="app-footer">
        <p>&copy; 2024 AI Calorie Analyzer. Powered by advanced computer vision and machine learning.</p>
      </footer>
    </div>
  );
}

export default App;
