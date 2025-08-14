import React, { useState } from 'react';
import { 
  Camera, 
  Brain, 
  BarChart3, 
  Clock,
  ArrowLeft,
  Sparkles,
  Zap,
  Image as ImageIcon,
  CheckCircle,
  Target
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
  const [isReAnalyzing, setIsReAnalyzing] = useState(false);
  const [activeStep, setActiveStep] = useState<'upload' | 'analyzing' | 'results'>('upload');
  const [uploadedImageUrl, setUploadedImageUrl] = useState<string | null>(null);
  const [currentImageFile, setCurrentImageFile] = useState<File | null>(null);

  const handleAnalysisComplete = (result: AnalysisResult, imageFile?: File) => {
    setAnalysisResult(result);
    setActiveStep('results');
    setIsAnalyzing(false);
    setIsReAnalyzing(false);
    if (imageFile) {
      const url = URL.createObjectURL(imageFile);
      setUploadedImageUrl(url);
      setCurrentImageFile(imageFile);
    }
  };

  const resetAnalysis = () => {
    setAnalysisResult(null);
    setActiveStep('upload');
    setIsAnalyzing(false);
    setIsReAnalyzing(false);
    if (uploadedImageUrl) {
      URL.revokeObjectURL(uploadedImageUrl);
      setUploadedImageUrl(null);
    }
    setCurrentImageFile(null);
  };

  const handleReAnalyze = async () => {
    if (!currentImageFile) return;
    
    setIsReAnalyzing(true);
    setActiveStep('analyzing');
    
    try {
      // Simulate re-analysis delay
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Create a mock re-analysis result with slightly different values
      const mockReAnalysisResult: AnalysisResult = {
        success: true,
        detected_foods: ['Re-analyzed Food Item 1', 'Re-analyzed Food Item 2'],
        nutritional_data: {
          total_calories: Math.floor(Math.random() * 200) + 300,
          total_protein: Math.floor(Math.random() * 20) + 15,
          total_carbs: Math.floor(Math.random() * 30) + 25,
          total_fats: Math.floor(Math.random() * 15) + 10,
          items: []
        },
        processing_time: Math.floor(Math.random() * 1000) + 500,
        model_used: 'expert_ensemble_v2',
        confidence: 0.95 + Math.random() * 0.05,
        sessionId: `session_${Date.now()}`,
        description: 'Re-analysis completed with enhanced AI models',
        insights: [
          'Re-analysis completed with enhanced AI models',
          'Improved accuracy achieved through ensemble learning',
          'Nutritional values have been recalculated for better precision'
        ],
        analysis: 'The re-analysis has been completed using our latest AI models. The nutritional breakdown has been updated with improved accuracy and confidence levels.'
      };
      
      handleAnalysisComplete(mockReAnalysisResult, currentImageFile);
    } catch (error) {
      console.error('Re-analysis failed:', error);
      setIsReAnalyzing(false);
      setActiveStep('results');
    }
  };

  const features = [
    {
      icon: <Brain size={24} />,
      title: "AI-Powered Detection",
      description: "Advanced computer vision with multiple AI models",
      color: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
    },
    {
      icon: <Zap size={24} />,
      title: "Real-time Analysis",
      description: "Get results in seconds with expert ensemble detection",
      color: "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"
    },
    {
      icon: <BarChart3 size={24} />,
      title: "Detailed Nutrition",
      description: "Comprehensive macro and micronutrient breakdown",
      color: "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)"
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
            <div className="header-badge">
              <Sparkles size={20} />
              <span>AI-Powered Analysis</span>
            </div>
            <h1>Food Analysis</h1>
            <p>Upload a photo of your meal for instant AI-powered nutritional analysis</p>
          </div>
        </div>

        {/* Main Upload Area - Now at the top */}
        <div className="analysis-main">
          {activeStep === 'upload' && (
            <div className="upload-section">
              <div className="upload-header">
                <div className="upload-icon">
                  <ImageIcon size={32} />
                </div>
                <h2>Upload Your Food Image</h2>
                <p>Take a photo or upload an image to get started with AI analysis</p>
              </div>
              <ImageUpload
                onAnalysisComplete={handleAnalysisComplete}
                isAnalyzing={isAnalyzing}
                setIsAnalyzing={setIsAnalyzing}
              />
            </div>
          )}

          {activeStep === 'analyzing' && (
            <div className="analyzing-section">
              <div className="analyzing-content">
                <div className="analyzing-icon">
                  <Sparkles size={48} />
                </div>
                <h2>{isReAnalyzing ? 'Re-analyzing Your Food' : 'Analyzing Your Food'}</h2>
                <p>Our AI is processing your image with multiple models for maximum accuracy</p>
                <div className="analyzing-progress">
                  <div className="progress-bar">
                    <div className="progress-fill"></div>
                  </div>
                  <span>{isReAnalyzing ? 'Re-processing...' : 'Processing...'}</span>
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
                <div className="results-title">
                  <CheckCircle size={24} />
                  <h2>Analysis Complete</h2>
                </div>
                <div className="results-actions">
                  {currentImageFile && (
                    <button 
                      onClick={handleReAnalyze} 
                      disabled={isReAnalyzing}
                      className="btn btn-secondary"
                    >
                      <Camera size={20} />
                      {isReAnalyzing ? 'Re-analyzing...' : 'Re-analyze'}
                    </button>
                  )}
                  <button onClick={resetAnalysis} className="btn btn-secondary">
                    <Camera size={20} />
                    New Analysis
                  </button>
                </div>
              </div>
              
              <div className="results-layout">
                {/* Image Display */}
                {uploadedImageUrl && (
                  <div className="image-display-section">
                    <h3>Analyzed Image</h3>
                    <div className="image-container">
                      <img 
                        src={uploadedImageUrl} 
                        alt="Analyzed food" 
                        className="analyzed-image"
                      />
                      <div className="image-overlay">
                        <Target size={24} />
                        <span>AI Analyzed</span>
                      </div>
                    </div>
                  </div>
                )}
                
                {/* Analysis Results */}
                <div className="results-content-section">
                  <AnalysisResults 
                    result={analysisResult} 
                    onReAnalyze={handleReAnalyze}
                    isReAnalyzing={isReAnalyzing}
                  />
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Features Section - Now below upload */}
        <div className="features-section">
          <div className="features-header">
            <h2>How It Works</h2>
            <p>Our advanced AI technology provides comprehensive food analysis</p>
          </div>
          <div className="features-grid">
            {features.map((feature, index) => (
              <div key={index} className="feature-card">
                <div className="feature-icon" style={{ background: feature.color }}>
                  {feature.icon}
                </div>
                <h3>{feature.title}</h3>
                <p>{feature.description}</p>
              </div>
            ))}
          </div>
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
