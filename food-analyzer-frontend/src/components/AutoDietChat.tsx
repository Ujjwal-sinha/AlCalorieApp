import React, { useState } from 'react';
import './AutoDietChat.css';

interface AutoDietChatProps {
  dietChatResponse: {
    answer: string;
    suggestions: string[];
    relatedTopics: string[];
    confidence: number;
  };
}

const AutoDietChat: React.FC<AutoDietChatProps> = ({ dietChatResponse }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const getConfidenceClass = (confidence: number) => {
    if (confidence >= 0.8) return 'high';
    if (confidence >= 0.6) return 'medium';
    return 'low';
  };

  const getConfidenceText = (confidence: number) => {
    if (confidence >= 0.8) return 'High Confidence';
    if (confidence >= 0.6) return 'Medium Confidence';
    return 'Low Confidence';
  };

  return (
    <div className="auto-diet-chat">
      <div className="auto-chat-header" onClick={() => setIsExpanded(!isExpanded)}>
        <div className="header-content">
          <div className="header-icon">🤖</div>
          <div className="header-text">
            <h3>AI Nutrition Assistant</h3>
            <p>Automatic meal analysis and nutrition advice</p>
          </div>
          <div className={`confidence-badge ${getConfidenceClass(dietChatResponse.confidence)}`}>
            <span className="confidence-text">{getConfidenceText(dietChatResponse.confidence)}</span>
            <span className="confidence-score">{(dietChatResponse.confidence * 100).toFixed(0)}%</span>
          </div>
        </div>
        <button 
          className={`expand-button ${isExpanded ? 'expanded' : ''}`}
          onClick={(e) => {
            e.stopPropagation();
            setIsExpanded(!isExpanded);
          }}
        >
          {isExpanded ? '▼' : '▶'}
        </button>
      </div>

      <div className={`auto-chat-content ${isExpanded ? 'expanded' : ''}`}>
        <div className="ai-response">
          <div className="response-text">
            {dietChatResponse.answer}
          </div>
        </div>

        {dietChatResponse.suggestions && dietChatResponse.suggestions.length > 0 && (
          <div className="suggestions-section">
            <h6>💡 Actionable Suggestions</h6>
            <div className="suggestions-list">
              {dietChatResponse.suggestions.map((suggestion, index) => (
                <div key={index} className="suggestion-item">
                  <span className="suggestion-icon">•</span>
                  <span className="suggestion-text">{suggestion}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {dietChatResponse.relatedTopics && dietChatResponse.relatedTopics.length > 0 && (
          <div className="topics-section">
            <h6>🔗 Related Topics</h6>
            <div className="topics-list">
              {dietChatResponse.relatedTopics.map((topic, index) => (
                <span key={index} className="topic-tag">
                  {topic}
                </span>
              ))}
            </div>
          </div>
        )}

        <div className="auto-chat-footer">
          <div className="footer-info">
            <span className="info-icon">ℹ️</span>
            <span>This analysis was automatically generated based on your uploaded meal image.</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AutoDietChat;
