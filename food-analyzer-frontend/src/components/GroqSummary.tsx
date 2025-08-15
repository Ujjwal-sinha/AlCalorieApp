import React from 'react';
import { Brain, Star, Lightbulb, Target } from 'lucide-react';
import './GroqSummary.css';

interface GroqSummaryProps {
  summary: string;
  healthScore: number;
  detectedFoods: string[];
}

const GroqSummary: React.FC<GroqSummaryProps> = ({ summary, healthScore, detectedFoods }) => {
  const getHealthScoreColor = (score: number) => {
    if (score >= 8) return '#10b981'; // Green
    if (score >= 6) return '#f59e0b'; // Yellow
    return '#ef4444'; // Red
  };

  const getHealthScoreLabel = (score: number) => {
    if (score >= 8) return 'Excellent';
    if (score >= 6) return 'Good';
    if (score >= 4) return 'Fair';
    return 'Poor';
  };

  return (
    <div className="groq-summary">
      <div className="summary-header">
        <div className="summary-title">
          <Brain size={24} className="summary-icon" />
          <h3>AI Nutrition Summary</h3>
          <div className="groq-badge">
            <Star size={16} />
            <span>GROQ AI</span>
          </div>
        </div>
      </div>

      <div className="summary-content">
        <div className="summary-main">
          <div className="summary-text">
            <Lightbulb size={20} className="insight-icon" />
            <p>{summary}</p>
          </div>
        </div>

        <div className="summary-details">
          <div className="health-score-section">
            <div className="score-header">
              <Target size={20} />
              <span>Nutritional Quality</span>
            </div>
            <div className="score-display">
              <div 
                className="score-circle"
                style={{ 
                  background: `conic-gradient(${getHealthScoreColor(healthScore)} ${healthScore * 36}deg, #e5e7eb ${healthScore * 36}deg)` 
                }}
              >
                <div className="score-inner">
                  <span className="score-number">{healthScore}</span>
                  <span className="score-max">/10</span>
                </div>
              </div>
              <div className="score-label">
                <span className="score-text">{getHealthScoreLabel(healthScore)}</span>
              </div>
            </div>
          </div>

          <div className="detected-foods-section">
            <h4>Detected Foods</h4>
            <div className="food-tags">
              {detectedFoods.map((food, index) => (
                <span key={index} className="food-tag">
                  {food}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default GroqSummary;
