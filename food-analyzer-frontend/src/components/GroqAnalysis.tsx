import React, { useState } from 'react';
import {
  Brain,
  Star,
  Lightbulb,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  Heart,
  Target,
  Clock,
  Award
} from 'lucide-react';
import './GroqAnalysis.css';

interface GroqAnalysisProps {
  analysis: {
    summary: string;
    detailedAnalysis: string;
    healthScore: number;
    recommendations: string[];
    dietaryConsiderations: string[];
  };
}

const GroqAnalysis: React.FC<GroqAnalysisProps> = ({ analysis }) => {
  const [isExpanded, setIsExpanded] = useState(false);

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
    <div className="groq-analysis">
      {/* Header */}
      <div className="groq-header">
        <div className="groq-title">
          <Brain size={24} className="groq-icon" />
          <h3>AI-Powered Nutrition Analysis</h3>
          <div className="groq-badge">
            <Star size={16} />
            <span>GROQ AI</span>
          </div>
        </div>
        <button
          className="expand-button"
          onClick={() => setIsExpanded(!isExpanded)}
        >
          {isExpanded ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
        </button>
      </div>

      {/* Health Score */}
      <div className="health-score-section">
        <div className="health-score-card">
          <div className="score-header">
            <Target size={20} />
            <span>Nutritional Quality Score</span>
          </div>
          <div className="score-display">
            <div 
              className="score-circle"
              style={{ 
                background: `conic-gradient(${getHealthScoreColor(analysis.healthScore)} ${analysis.healthScore * 36}deg, #e5e7eb ${analysis.healthScore * 36}deg)` 
              }}
            >
              <div className="score-inner">
                <span className="score-number">{analysis.healthScore}</span>
                <span className="score-max">/10</span>
              </div>
            </div>
            <div className="score-label">
              <span className="score-text">{getHealthScoreLabel(analysis.healthScore)}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Executive Summary */}
      <div className="summary-section">
        <div className="section-header">
          <Lightbulb size={20} />
          <h4>Executive Summary</h4>
        </div>
        <div className="summary-content">
          <p>{analysis.summary}</p>
        </div>
      </div>

      {/* Recommendations */}
      <div className="recommendations-section">
        <div className="section-header">
          <Heart size={20} />
          <h4>Health Recommendations</h4>
        </div>
        <div className="recommendations-list">
          {analysis.recommendations.map((recommendation, index) => (
            <div key={index} className="recommendation-item">
              <div className="recommendation-icon">
                <Award size={16} />
              </div>
              <span>{recommendation}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Dietary Considerations */}
      <div className="dietary-section">
        <div className="section-header">
          <AlertTriangle size={20} />
          <h4>Dietary Considerations</h4>
        </div>
        <div className="dietary-list">
          {analysis.dietaryConsiderations.map((consideration, index) => (
            <div key={index} className="dietary-item">
              <div className="dietary-icon">
                <Clock size={16} />
              </div>
              <span>{consideration}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Detailed Analysis (Expandable) */}
      {isExpanded && (
        <div className="detailed-analysis-section">
          <div className="section-header">
            <Brain size={20} />
            <h4>Detailed Analysis</h4>
          </div>
          <div className="detailed-content">
            <pre className="analysis-text">{analysis.detailedAnalysis}</pre>
          </div>
        </div>
      )}
    </div>
  );
};

export default GroqAnalysis;
