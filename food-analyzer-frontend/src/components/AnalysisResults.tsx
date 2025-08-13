import React from 'react';
import { CheckCircle, AlertCircle, Zap, Target, TrendingUp } from 'lucide-react';
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

  const { nutritional_data } = result;
  const totalCalories = nutritional_data.total_calories;
  const proteinPercentage = Math.round((nutritional_data.total_protein * 4 / totalCalories) * 100);
  const carbsPercentage = Math.round((nutritional_data.total_carbs * 4 / totalCalories) * 100);
  const fatsPercentage = Math.round((nutritional_data.total_fats * 9 / totalCalories) * 100);

  return (
    <div className="analysis-results">
      <div className="results-header">
        <CheckCircle size={32} className="success-icon" />
        <h2>Analysis Complete</h2>
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
              <span className="stat-value">{nutritional_data.total_protein}g</span>
              <span className="stat-label">Protein ({proteinPercentage}%)</span>
            </div>
            <div className="stat-item carbs">
              <span className="stat-value">{nutritional_data.total_carbs}g</span>
              <span className="stat-label">Carbs ({carbsPercentage}%)</span>
            </div>
            <div className="stat-item fats">
              <span className="stat-value">{nutritional_data.total_fats}g</span>
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
            {nutritional_data.items.map((item, index) => (
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
                  <span>{item.calories} cal</span>
                  <span>{item.protein}g protein</span>
                  <span>{item.carbs}g carbs</span>
                  <span>{item.fats}g fats</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Analysis Text */}
      <div className="result-card analysis-text">
        <h3>
          <TrendingUp size={20} />
          Detailed Analysis
        </h3>
        <div className="analysis-content">
          <pre>{result.analysis}</pre>
        </div>
      </div>
    </div>
  );
};

export default AnalysisResults;