import React from 'react';
import {
  CheckCircle,
  AlertCircle,
  Clock,
  Brain,
  Lightbulb,
  Zap,
  Apple,
  BarChart3,
  TrendingUp,
  Target,
  RefreshCw,
  Globe,
  Activity
} from 'lucide-react';
import type { AnalysisResult } from '../types';
import './AnalysisResults.css';

interface AnalysisResultsProps {
  result: AnalysisResult;
  onReAnalyze?: () => void;
  isReAnalyzing?: boolean;
}

// Model Detection Breakdown Component
const ModelDetectionBreakdown: React.FC<{ modelInfo: AnalysisResult['model_info'] }> = ({ modelInfo }) => {
  if (!modelInfo) return null;

  const { model_performance, detailed_detections } = modelInfo;
  
  // Calculate total detections and success rate
  const totalDetections = Object.values(model_performance).reduce((sum, model) => 
    sum + (model.success ? model.detection_count : 0), 0
  );
  
  const successfulModels = Object.values(model_performance).filter(model => model.success).length;
  const totalModels = Object.keys(model_performance).length;
  const successRate = totalModels > 0 ? (successfulModels / totalModels) * 100 : 0;

  return (
    <div className="model-detection-breakdown">
      <h3>Expert Multi-Model Detection Results</h3>
      
      {/* Model Performance Summary */}
      <div className="model-performance-summary">
        <h4>
          <Globe size={16} />
          Model Performance Summary
        </h4>
        <div className="performance-grid">
          <div className="performance-item">
            <span className="performance-label">Detection Method</span>
            <span className="performance-value">Comprehensive Ensemble</span>
          </div>
          <div className="performance-item">
            <span className="performance-label">Total Detections</span>
            <span className="performance-value">{totalDetections}</span>
          </div>
          <div className="performance-item">
            <span className="performance-label">Success Rate</span>
            <span className="performance-value success-rate">
              <CheckCircle size={16} />
              {successRate.toFixed(0)}%
            </span>
          </div>
        </div>
      </div>

      {/* Model Breakdown */}
      <div className="model-breakdown">
        <h4>
          <BarChart3 size={16} />
          Model Breakdown
        </h4>
        <div className="model-list">
          {Object.entries(model_performance).map(([modelName, performance]) => (
            <div key={modelName} className={`model-item ${performance.success ? 'success' : 'failed'}`}>
              <div className="model-name">
                {modelName.toUpperCase()}
              </div>
              <div className="model-stats">
                <span className="detection-count">{performance.detection_count}</span>
                {performance.success ? (
                  <CheckCircle size={14} className="status-icon success" />
                ) : (
                  <AlertCircle size={14} className="status-icon failed" />
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Detailed Food Analysis */}
      {detailed_detections && detailed_detections.length > 0 && (
        <div className="detailed-food-analysis">
          <h4>
            <Activity size={16} />
            Detailed Food Analysis
          </h4>
          <div className="food-detection-list">
            {detailed_detections.map((detection, index) => (
              <div key={index} className="food-detection-item">
                <div className="food-number">{index + 1}.</div>
                <div className="food-name">{detection.food}</div>
                <div className="food-methods">
                  {detection.methods.join(', ')}
                </div>
                <div className="food-confidence">
                  {(detection.avg_confidence * 100).toFixed(0)}%
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

const AnalysisResults: React.FC<AnalysisResultsProps> = ({ 
  result, 
  onReAnalyze, 
  isReAnalyzing = false 
}) => {
  // Handle case where no food is detected
  if (!result.success) {
    return (
      <div className="analysis-results">
        <div className="results-header">
          <div className="status-indicator error">
            <AlertCircle size={24} />
            <span>Analysis Complete</span>
          </div>
          {onReAnalyze && (
            <button 
              onClick={onReAnalyze} 
              disabled={isReAnalyzing}
              className="re-analyze-btn"
            >
              <RefreshCw size={16} className={isReAnalyzing ? 'spinning' : ''} />
              {isReAnalyzing ? 'Re-analyzing...' : 'Re-analyze'}
            </button>
          )}
        </div>

        <div className="results-content">
          <div className="error-message">
            <h3>No Food Items Detected</h3>
            <p>{result.analysis || 'The AI models were unable to detect any food items in the image with sufficient confidence.'}</p>

            <div className="suggestions">
              <h4>Suggestions for better results:</h4>
              <ul>
                <li>Ensure the image contains clear, well-lit food items</li>
                <li>Make sure the food is the main focus of the image</li>
                <li>Try taking the photo from a closer distance</li>
                <li>Ensure good lighting conditions</li>
                <li>Avoid blurry or low-quality images</li>
              </ul>
            </div>

            {result.insights && result.insights.length > 0 && (
              <div className="insights-section">
                <h4>AI Insights:</h4>
                <ul className="insights-list">
                  {result.insights.map((insight, index) => (
                    <li key={index} className="insight-item">
                      <span className="insight-bullet">•</span>
                      <span className="insight-text">{insight}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  // Handle successful detection
  const nutritionalData = result.nutritional_data || {
    total_calories: 0,
    total_protein: 0,
    total_carbs: 0,
    total_fats: 0,
    items: []
  };

  const detectedFoods = result.detected_foods || [];
  const insights = result.insights || [];

  return (
    <div className="analysis-results">
      <div className="results-header">
        <div className="status-indicator success">
          <CheckCircle size={24} />
          <span>Expert Analysis Complete</span>
        </div>
        {onReAnalyze && (
          <button 
            onClick={onReAnalyze} 
            disabled={isReAnalyzing}
            className="re-analyze-btn"
          >
            <RefreshCw size={16} className={isReAnalyzing ? 'spinning' : ''} />
            {isReAnalyzing ? 'Re-analyzing...' : 'Re-analyze'}
          </button>
        )}
      </div>

      <div className="results-content">
        {/* Model Detection Breakdown */}
        {result.model_info && (
          <ModelDetectionBreakdown modelInfo={result.model_info} />
        )}

        {/* Nutrition Summary */}
        <div className="nutrition-summary">
          <h3>Nutrition Summary</h3>
          <div className="nutrition-grid">
            <div className="nutrition-item">
              <Zap size={20} />
              <div className="nutrition-content">
                <span className="value">{nutritionalData.total_calories}</span>
                <span className="label">Calories</span>
              </div>
            </div>
            <div className="nutrition-item">
              <Apple size={20} />
              <div className="nutrition-content">
                <span className="value">{nutritionalData.total_protein}g</span>
                <span className="label">Protein</span>
              </div>
            </div>
            <div className="nutrition-item">
              <BarChart3 size={20} />
              <div className="nutrition-content">
                <span className="value">{nutritionalData.total_carbs}g</span>
                <span className="label">Carbs</span>
              </div>
            </div>
            <div className="nutrition-item">
              <TrendingUp size={20} />
              <div className="nutrition-content">
                <span className="value">{nutritionalData.total_fats}g</span>
                <span className="label">Fats</span>
              </div>
            </div>
          </div>
        </div>

        {/* Detected Foods */}
        {detectedFoods.length > 0 && (
          <div className="detected-foods">
            <h3>Detected Food Items</h3>
            <div className="food-tags">
              {detectedFoods.map((food, index) => (
                <span key={index} className="detected-food-tag">
                  {food}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Expert Details */}
        <div className="expert-details">
          <h3>Processing Information</h3>
          <div className="detail-item">
            <Clock size={16} />
            <span className="detail-label">Processing Time:</span>
            <span className="detail-value">{result.processing_time || 0}ms</span>
          </div>
          <div className="detail-item">
            <Brain size={16} />
            <span className="detail-label">AI Model Used:</span>
            <span className="detail-value">{result.model_used || 'expert_ensemble'}</span>
          </div>
          {result.confidence && (
            <div className="detail-item">
              <Target size={16} />
              <span className="detail-label">Overall Confidence:</span>
              <span className="detail-value">{Math.round((result.confidence || 0) * 100)}%</span>
            </div>
          )}
          {result.sessionId && (
            <div className="detail-item">
              <span className="detail-label">Session ID:</span>
              <span className="detail-value session-id">{result.sessionId}</span>
            </div>
          )}
        </div>

        {/* AI Insights */}
        {insights.length > 0 && (
          <div className="insights-section">
            <h3>AI Insights</h3>
            <ul className="insights-list">
              {insights.map((insight, index) => (
                <li key={index} className="insight-item">
                  <Lightbulb size={16} />
                  <div className="content">
                    <div className="title">AI Insight</div>
                    <div className="description">{insight}</div>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Detailed Analysis */}
        {result.analysis && (
          <div className="detailed-analysis">
            <h3>Detailed Analysis</h3>
            <div className="analysis-text">
              {result.analysis.split('\n').map((paragraph, index) => (
                <p key={index}>{paragraph}</p>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AnalysisResults;