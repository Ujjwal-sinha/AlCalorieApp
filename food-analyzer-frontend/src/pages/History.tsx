import React, { useState, useEffect } from 'react';
import { 
  History as HistoryIcon, 
  Search, 
  Calendar,
  BarChart3,
  Trash2,
  Eye,
  Download,
  ArrowLeft
} from 'lucide-react';
import { Link } from 'react-router-dom';
import Navigation from '../components/Navigation';
import type { HistoryEntry } from '../types';
import { HistoryService } from '../services/HistoryService';
import './History.css';

const History: React.FC = () => {
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [filteredHistory, setFilteredHistory] = useState<HistoryEntry[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedFilter, setSelectedFilter] = useState('all');
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    loadHistory();
  }, []);

  useEffect(() => {
    filterHistory();
  }, [history, searchTerm, selectedFilter]);

  const loadHistory = () => {
    const historyService = HistoryService.getInstance();
    const savedHistory = historyService.getHistory();
    setHistory(savedHistory);
    setIsLoading(false);
  };

  const filterHistory = () => {
    let filtered = history;

    // Apply search filter
    if (searchTerm) {
      filtered = filtered.filter(entry => 
        entry.analysis_result.description?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        entry.analysis_result.detected_foods?.some(food => 
          food.toLowerCase().includes(searchTerm.toLowerCase())
        )
      );
    }

    // Apply date filter
    switch (selectedFilter) {
      case 'today':
        filtered = filtered.filter(entry => {
          const today = new Date();
          const entryDate = new Date(entry.timestamp);
          return entryDate.toDateString() === today.toDateString();
        });
        break;
      case 'week':
        filtered = filtered.filter(entry => {
          const weekAgo = new Date();
          weekAgo.setDate(weekAgo.getDate() - 7);
          return new Date(entry.timestamp) >= weekAgo;
        });
        break;
      case 'month':
        filtered = filtered.filter(entry => {
          const monthAgo = new Date();
          monthAgo.setMonth(monthAgo.getMonth() - 1);
          return new Date(entry.timestamp) >= monthAgo;
        });
        break;
      default:
        break;
    }

    setFilteredHistory(filtered);
  };

  const deleteEntry = (id: string) => {
    // Remove from localStorage and update state
    const updatedHistory = history.filter(entry => entry.id !== id);
    localStorage.setItem('analysisHistory', JSON.stringify(updatedHistory));
    setHistory(updatedHistory);
  };

  const formatDate = (date: Date) => {
    return new Intl.DateTimeFormat('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    }).format(date);
  };

  const getCalories = (entry: HistoryEntry) => {
    return entry.analysis_result.nutritional_data?.total_calories || 0;
  };

  const getFoodCount = (entry: HistoryEntry) => {
    return entry.analysis_result.detected_foods?.length || 0;
  };

  if (isLoading) {
    return (
      <div className="history-page">
        <Navigation />
        <div className="loading-state">
          <HistoryIcon size={48} />
          <h2>Loading History...</h2>
        </div>
      </div>
    );
  }

  return (
    <div className="history-page">
      <Navigation />
      
      <div className="history-content">
        {/* Header */}
        <div className="history-header">
          <Link to="/dashboard" className="back-button">
            <ArrowLeft size={20} />
            Back to Dashboard
          </Link>
          <div className="header-content">
            <h1>Analysis History</h1>
            <p>View and manage your past food analyses</p>
          </div>
        </div>

        {/* Stats */}
        <div className="history-stats">
          <div className="stat-card">
            <HistoryIcon size={24} />
            <div className="stat-content">
              <h3>{history.length}</h3>
              <p>Total Analyses</p>
            </div>
          </div>
          <div className="stat-card">
            <BarChart3 size={24} />
            <div className="stat-content">
              <h3>{history.reduce((sum, entry) => sum + getCalories(entry), 0)}</h3>
              <p>Total Calories</p>
            </div>
          </div>
          <div className="stat-card">
            <Calendar size={24} />
            <div className="stat-content">
              <h3>{history.length > 0 ? formatDate(new Date(history[0].timestamp)).split(',')[0] : 'N/A'}</h3>
              <p>Last Analysis</p>
            </div>
          </div>
        </div>

        {/* Filters */}
        <div className="filters-section">
          <div className="search-box">
            <Search size={20} />
            <input
              type="text"
              placeholder="Search analyses..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
          <div className="filter-buttons">
            <button
              className={`filter-btn ${selectedFilter === 'all' ? 'active' : ''}`}
              onClick={() => setSelectedFilter('all')}
            >
              All
            </button>
            <button
              className={`filter-btn ${selectedFilter === 'today' ? 'active' : ''}`}
              onClick={() => setSelectedFilter('today')}
            >
              Today
            </button>
            <button
              className={`filter-btn ${selectedFilter === 'week' ? 'active' : ''}`}
              onClick={() => setSelectedFilter('week')}
            >
              This Week
            </button>
            <button
              className={`filter-btn ${selectedFilter === 'month' ? 'active' : ''}`}
              onClick={() => setSelectedFilter('month')}
            >
              This Month
            </button>
          </div>
        </div>

        {/* History List */}
        <div className="history-list">
          {filteredHistory.length === 0 ? (
            <div className="empty-state">
              <HistoryIcon size={64} />
              <h2>No analyses found</h2>
              <p>Start analyzing your food to see your history here</p>
              <Link to="/analysis" className="btn btn-primary">
                Start Analysis
              </Link>
            </div>
          ) : (
            filteredHistory.map((entry) => (
              <div key={entry.id} className="history-item">
                <div className="item-image">
                  {entry.image_url ? (
                    <img src={entry.image_url} alt="Food analysis" />
                  ) : (
                    <div className="placeholder-image">
                      <BarChart3 size={24} />
                    </div>
                  )}
                </div>
                <div className="item-content">
                  <div className="item-header">
                    <h3>{entry.analysis_result.description || 'Food Analysis'}</h3>
                    <span className="item-date">{formatDate(entry.timestamp)}</span>
                  </div>
                  <div className="item-details">
                    <div className="detail">
                      <span className="label">Calories:</span>
                      <span className="value">{getCalories(entry)} cal</span>
                    </div>
                    <div className="detail">
                      <span className="label">Foods:</span>
                      <span className="value">{getFoodCount(entry)} items</span>
                    </div>
                    <div className="detail">
                      <span className="label">Confidence:</span>
                      <span className="value">
                        {Math.round((entry.analysis_result.confidence || 0) * 100)}%
                      </span>
                    </div>
                  </div>
                  <div className="item-foods">
                    {entry.analysis_result.detected_foods?.slice(0, 3).map((food, index) => (
                      <span key={index} className="food-tag">
                        {food}
                      </span>
                    ))}
                    {entry.analysis_result.detected_foods && entry.analysis_result.detected_foods.length > 3 && (
                      <span className="food-tag more">
                        +{entry.analysis_result.detected_foods.length - 3} more
                      </span>
                    )}
                  </div>
                </div>
                <div className="item-actions">
                  <button className="action-btn" title="View Details">
                    <Eye size={16} />
                  </button>
                  <button className="action-btn" title="Download Report">
                    <Download size={16} />
                  </button>
                  <button 
                    className="action-btn delete" 
                    title="Delete"
                    onClick={() => deleteEntry(entry.id)}
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default History;
