import React from 'react';
import { Clock, Eye, Calendar } from 'lucide-react';
import type { HistoryEntry } from '../types';

interface HistoryViewProps {
  history: HistoryEntry[];
  onSelectEntry: (entry: HistoryEntry) => void;
}

const HistoryView: React.FC<HistoryViewProps> = ({ history, onSelectEntry }) => {
  const formatDate = (date: Date) => {
    return new Intl.DateTimeFormat('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    }).format(date);
  };

  const formatCalories = (calories: number) => {
    return calories.toLocaleString();
  };

  if (history.length === 0) {
    return (
      <div className="history-view empty">
        <div className="empty-state">
          <Calendar size={64} className="empty-icon" />
          <h2>No Analysis History</h2>
          <p>Your food analysis history will appear here once you start analyzing images.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="history-view">
      <div className="history-header">
        <h2>
          <Clock size={24} />
          Analysis History
        </h2>
        <p>View your previous food analyses and nutritional data</p>
      </div>

      <div className="history-list">
        {history.map((entry) => (
          <div key={entry.id} className="history-item">
            <div className="history-content">
              {entry.image_url && (
                <div className="history-image">
                  <img src={entry.image_url} alt="Food analysis" />
                </div>
              )}
              
              <div className="history-details">
                <div className="history-meta">
                  <span className="timestamp">{formatDate(entry.timestamp)}</span>
                  <span className={`status ${entry.analysis_result.success ? 'success' : 'error'}`}>
                    {entry.analysis_result.success ? 'Success' : 'Failed'}
                  </span>
                </div>
                
                <h3 className="description">{entry.analysis_result.description}</h3>
                
                {entry.analysis_result.success && (
                  <div className="nutrition-summary">
                    <div className="nutrition-item">
                      <span className="label">Calories:</span>
                      <span className="value">
                        {formatCalories(entry.analysis_result.nutritional_data.total_calories)}
                      </span>
                    </div>
                    <div className="nutrition-item">
                      <span className="label">Items:</span>
                      <span className="value">
                        {entry.analysis_result.nutritional_data.items.length}
                      </span>
                    </div>
                    <div className="nutrition-item">
                      <span className="label">Protein:</span>
                      <span className="value">
                        {entry.analysis_result.nutritional_data.total_protein}g
                      </span>
                    </div>
                  </div>
                )}
                
                {entry.context && (
                  <div className="context">
                    <span className="context-label">Context:</span>
                    <span className="context-text">{entry.context}</span>
                  </div>
                )}
              </div>
            </div>
            
            <div className="history-actions">
              <button
                className="action-button view"
                onClick={() => onSelectEntry(entry)}
                title="View Details"
              >
                <Eye size={16} />
                View
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default HistoryView;