import React from 'react';
import { CheckCircle, AlertCircle, Zap, Target, TrendingUp, Clock, Brain, Lightbulb } from 'lucide-react';
import type { AnalysisResult } from '../types';

interface AnalysisResultsProps {
  result: AnalysisResult;
}

const AnalysisResults: React.FC<AnalysisResultsProps> = ({ result }) => {
  if (!result.success) {
    return (
      <div className="analysis-results error">
        <div className="error-card">
          <AlertCircle size={48} className="error-icon" />
          <h2>Analysis Failed</h2>
          <p>{result.error || 'Unable to analyze the image'}</p>
          <p className="error-description">{result.description}</p>
        </div>
      </div>
    );
  }

  // Ensure nutritional_data exists with default values
  const nutritional_data = result.nutritional_data || {
    total_calories: 0,
    total_protein: 0,
    total_carbs: 0,
    total_fats: 0,
    items: []
  };

  const totalCalories = nutritional_data.total_calories || 0;
  
  // Calculate percentages safely
  const proteinPercentage = totalCalories > 0 ? Math.round((nutritional_data.total_protein * 4 / totalCalories) * 100) : 0;
  const carbsPercentage = totalCalories > 0 ? Math.round((nutritional_data.total_carbs * 4 / totalCalories) * 100) : 0;
  const fatsPercentage = totalCalories > 0 ? Math.round((nutritional_data.total_fats * 9 / totalCalories) * 100) : 0;

  return (
    <div className="analysis-results">
      <div className="results-header">
        <CheckCircle size={32} className="success-icon" />
        <h2>Expert Analysis Complete</h2>
        <p>{result.description}</p>
      </div>

      <div className="results-grid">
        {/* Nutritional Summary */}
        <div className="result-card nutrition-summary">
          <h3>
            <Zap size={20} />
            Nutritional Summary
          </h3>
          <div className="nutrition-stats">
            <div className="stat-item calories">
              <span className="stat-value">{totalCalories}</span>
              <span className="stat-label">Calories</span>
            </div>
            <div className="stat-item protein">
              <span className="stat-value">{nutritional_data.total_protein || 0}g</span>
              <span className="stat-label">Protein ({proteinPercentage}%)</span>
            </div>
            <div className="stat-item carbs">
              <span className="stat-value">{nutritional_data.total_carbs || 0}g</span>
              <span className="stat-label">Carbs ({carbsPercentage}%)</span>
            </div>
            <div className="stat-item fats">
              <span className="stat-value">{nutritional_data.total_fats || 0}g</span>
              <span className="stat-label">Fats ({fatsPercentage}%)</span>
            </div>
          </div>
        </div>

        {/* Food Items */}
        <div className="result-card food-items">
          <h3>
            <Target size={20} />
            Detected Food Items
          </h3>
          <div className="food-list">
            {nutritional_data.items && nutritional_data.items.length > 0 ? (
              nutritional_data.items.map((item, index) => (
                <div key={index} className="food-item">
                  <div className="food-info">
                    <span className="food-name">{item.name}</span>
                    {item.confidence && (
                      <span className="confidence">
                        {Math.round(item.confidence * 100)}% confidence
                      </span>
                    )}
                  </div>
                  <div className="food-nutrition">
                    <span>{item.calories || 0} cal</span>
                    <span>{item.protein || 0}g protein</span>
                    <span>{item.carbs || 0}g carbs</span>
                    <span>{item.fats || 0}g fats</span>
                  </div>
                </div>
              ))
            ) : (
              <div className="no-items">
                <p>No food items detected</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Expert Analysis Details */}
      <div className="expert-details">
        {/* Processing Information */}
        {(result.processing_time || result.model_used) && (
          <div className="result-card processing-info">
            <h3>
              <Clock size={20} />
              Processing Information
            </h3>
            <div className="processing-details">
              {result.processing_time && (
                <div className="detail-item">
                  <span className="detail-label">Processing Time:</span>
                  <span className="detail-value">{result.processing_time}ms</span>
                </div>
              )}
              {result.model_used && (
                <div className="detail-item">
                  <span className="detail-label">AI Model Used:</span>
                  <span className="detail-value">{result.model_used}</span>
                </div>
              )}
              {result.confidence && (
                <div className="detail-item">
                  <span className="detail-label">Overall Confidence:</span>
                  <span className="detail-value">{Math.round(result.confidence * 100)}%</span>
                </div>
              )}
              {result.sessionId && (
                <div className="detail-item">
                  <span className="detail-label">Session ID:</span>
                  <span className="detail-value session-id">{result.sessionId}</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* AI Insights */}
        {result.insights && result.insights.length > 0 && (
          <div className="result-card insights">
            <h3>
              <Lightbulb size={20} />
              AI Insights
            </h3>
            <div className="insights-list">
              {result.insights.map((insight, index) => (
                <div key={index} className="insight-item">
                  <span className="insight-bullet">•</span>
                  <span className="insight-text">{insight}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Detected Foods List */}
        {result.detected_foods && result.detected_foods.length > 0 && (
          <div className="result-card detected-foods">
            <h3>
              <Brain size={20} />
              Raw Detections
            </h3>
            <div className="detected-foods-list">
              {result.detected_foods.map((food, index) => (
                <span key={index} className="detected-food-tag">
                  {food}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Analysis Text */}
      <div className="result-card analysis-text">
        <h3>
          <TrendingUp size={20} />
          Detailed Analysis
        </h3>
        <div className="analysis-content">
          <pre>{result.analysis || 'No analysis available'}</pre>
        </div>
      </div>
    </div>
  );
};

export default AnalysisResults;