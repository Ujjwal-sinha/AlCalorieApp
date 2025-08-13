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
  Target
} from 'lucide-react';
import type { AnalysisResult } from '../types';
import './AnalysisResults.css';

interface AnalysisResultsProps {
  result: AnalysisResult;
}

const AnalysisResults: React.FC<AnalysisResultsProps> = ({ result }) => {
  // Handle case where no food is detected
  if (!result.success) {
    return (
      <div className="analysis-results">
        <div className="results-header">
          <div className="status-indicator error">
            <AlertCircle size={24} />
            <span>Analysis Complete</span>
          </div>
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
      </div>

      <div className="results-content">
        {/* Nutrition Summary */}
        <div className="nutrition-summary">
          <h3>Nutrition Summary</h3>
          <div className="nutrition-grid">
            <div className="nutrition-item">
              <Zap size={20} />
              <div className="nutrition-content">
                <span className="nutrition-value">{nutritionalData.total_calories}</span>
                <span className="nutrition-label">Calories</span>
              </div>
            </div>
            <div className="nutrition-item">
              <Apple size={20} />
              <div className="nutrition-content">
                <span className="nutrition-value">{nutritionalData.total_protein}g</span>
                <span className="nutrition-label">Protein</span>
              </div>
            </div>
            <div className="nutrition-item">
              <BarChart3 size={20} />
              <div className="nutrition-content">
                <span className="nutrition-value">{nutritionalData.total_carbs}g</span>
                <span className="nutrition-label">Carbs</span>
              </div>
            </div>
            <div className="nutrition-item">
              <TrendingUp size={20} />
              <div className="nutrition-content">
                <span className="nutrition-value">{nutritionalData.total_fats}g</span>
                <span className="nutrition-label">Fats</span>
              </div>
            </div>
          </div>
        </div>

        {/* Detected Foods */}
        {detectedFoods.length > 0 && (
          <div className="detected-foods">
            <h3>Detected Food Items</h3>
            <div className="detected-foods-list">
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
                  <span className="insight-text">{insight}</span>
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