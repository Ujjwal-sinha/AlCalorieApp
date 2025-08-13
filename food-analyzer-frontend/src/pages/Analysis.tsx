import React, { useState } from 'react';
import { 
  Camera, 
  Upload, 
  Brain, 
  BarChart3, 
  Clock,
  ArrowLeft,
  Sparkles,
  Zap
} from 'lucide-react';
import { Link } from 'react-router-dom';
import Navigation from '../components/Navigation';
import ImageUpload from '../components/ImageUpload';
import AnalysisResults from '../components/AnalysisResults';
import type { AnalysisResult } from '../types';
import './Analysis.css';

const Analysis: React.FC = () => {
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [activeStep, setActiveStep] = useState<'upload' | 'analyzing' | 'results'>('upload');

  const handleAnalysisComplete = (result: AnalysisResult) => {
    setAnalysisResult(result);
    setActiveStep('results');
    setIsAnalyzing(false);
  };

  const handleStartAnalysis = () => {
    setIsAnalyzing(true);
    setActiveStep('analyzing');
  };

  const resetAnalysis = () => {
    setAnalysisResult(null);
    setActiveStep('upload');
    setIsAnalyzing(false);
  };

  const features = [
    {
      icon: <Brain size={24} />,
      title: "AI-Powered Detection",
      description: "Advanced computer vision with multiple AI models"
    },
    {
      icon: <Zap size={24} />,
      title: "Real-time Analysis",
      description: "Get results in seconds with expert ensemble detection"
    },
    {
      icon: <BarChart3 size={24} />,
      title: "Detailed Nutrition",
      description: "Comprehensive macro and micronutrient breakdown"
    }
  ];

  return (
    <div className="analysis-page">
      <Navigation />
      
      <div className="analysis-content">
        {/* Header */}
        <div className="analysis-header">
          <Link to="/dashboard" className="back-button">
            <ArrowLeft size={20} />
            Back to Dashboard
          </Link>
          <div className="header-content">
            <h1>Food Analysis</h1>
            <p>Upload a photo of your meal for instant AI-powered nutritional analysis</p>
          </div>
        </div>

        {/* Features */}
        <div className="features-section">
          <div className="features-grid">
            {features.map((feature, index) => (
              <div key={index} className="feature-card">
                <div className="feature-icon">
                  {feature.icon}
                </div>
                <h3>{feature.title}</h3>
                <p>{feature.description}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Main Analysis Area */}
        <div className="analysis-main">
          {activeStep === 'upload' && (
            <div className="upload-section">
              <div className="upload-header">
                <h2>Upload Your Food Image</h2>
                <p>Take a photo or upload an image to get started</p>
              </div>
              <ImageUpload
                onAnalysisComplete={handleAnalysisComplete}
                isAnalyzing={isAnalyzing}
                setIsAnalyzing={setIsAnalyzing}
                onStartAnalysis={handleStartAnalysis}
              />
            </div>
          )}

          {activeStep === 'analyzing' && (
            <div className="analyzing-section">
              <div className="analyzing-content">
                <div className="analyzing-icon">
                  <Sparkles size={48} />
                </div>
                <h2>Analyzing Your Food</h2>
                <p>Our AI is processing your image with multiple models for maximum accuracy</p>
                <div className="analyzing-progress">
                  <div className="progress-bar">
                    <div className="progress-fill"></div>
                  </div>
                  <span>Processing...</span>
                </div>
                <div className="analyzing-steps">
                  <div className="step">
                    <Brain size={20} />
                    <span>AI Detection</span>
                  </div>
                  <div className="step">
                    <BarChart3 size={20} />
                    <span>Nutrition Analysis</span>
                  </div>
                  <div className="step">
                    <Clock size={20} />
                    <span>Results Ready</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeStep === 'results' && analysisResult && (
            <div className="results-section">
              <div className="results-header">
                <h2>Analysis Complete</h2>
                <button onClick={resetAnalysis} className="btn btn-secondary">
                  <Camera size={20} />
                  New Analysis
                </button>
              </div>
              <AnalysisResults result={analysisResult} />
            </div>
          )}
        </div>

        {/* Tips Section */}
        <div className="tips-section">
          <div className="tips-header">
            <h2>Tips for Better Results</h2>
            <p>Follow these guidelines for more accurate analysis</p>
          </div>
          <div className="tips-grid">
            <div className="tip-card">
              <div className="tip-icon">📸</div>
              <h3>Good Lighting</h3>
              <p>Ensure your food is well-lit for better AI recognition</p>
            </div>
            <div className="tip-card">
              <div className="tip-icon">🎯</div>
              <h3>Clear Focus</h3>
              <p>Keep the camera steady and focus on the main food items</p>
            </div>
            <div className="tip-card">
              <div className="tip-icon">📏</div>
              <h3>Proper Distance</h3>
              <p>Take photos from a reasonable distance to capture the full meal</p>
            </div>
            <div className="tip-card">
              <div className="tip-icon">🍽️</div>
              <h3>Complete View</h3>
              <p>Include all food items in the frame for comprehensive analysis</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Analysis;
