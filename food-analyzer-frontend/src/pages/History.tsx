import React, { useState } from 'react';
import { 
  ArrowLeft, 
  Search, 
  Calendar,
  Clock,
  Camera,
  Eye,
  Download,
  Trash2,
  BarChart3,
  TrendingUp,
  Target,
  Apple,
  Zap
} from 'lucide-react';
import { Link } from 'react-router-dom';
import Navigation from '../components/Navigation';
import './History.css';

// Mock data for history
const mockHistoryData = {
  totalAnalyses: 156,
  thisWeek: 23,
  thisMonth: 89,
  averageCalories: 1850,
  totalCalories: 288750,
  weeklyTrend: [12, 15, 18, 14, 20, 16, 23],
  monthlyTrend: [89, 92, 87, 95, 88, 91, 94, 89, 92, 87, 95, 88],
  analyses: [
    {
      id: 1,
      date: '2024-01-15',
      time: '12:30 PM',
      image: '/api/placeholder/150/150',
      foods: ['Grilled Chicken', 'Quinoa', 'Mixed Vegetables', 'Olive Oil'],
      totalCalories: 485,
      protein: 35,
      carbs: 45,
      fats: 18,
      confidence: 94,
      model: 'expert_ensemble',
      sessionId: 'sess_001_20240115_1230'
    },
    {
      id: 2,
      date: '2024-01-14',
      time: '7:45 PM',
      image: '/api/placeholder/150/150',
      foods: ['Salmon', 'Brown Rice', 'Asparagus', 'Lemon'],
      totalCalories: 520,
      protein: 42,
      carbs: 38,
      fats: 22,
      confidence: 96,
      model: 'expert_ensemble',
      sessionId: 'sess_002_20240114_1945'
    },
    {
      id: 3,
      date: '2024-01-14',
      time: '1:15 PM',
      image: '/api/placeholder/150/150',
      foods: ['Turkey Sandwich', 'Apple', 'Greek Yogurt', 'Nuts'],
      totalCalories: 420,
      protein: 28,
      carbs: 52,
      fats: 15,
      confidence: 91,
      model: 'expert_ensemble',
      sessionId: 'sess_003_20240114_1315'
    },
    {
      id: 4,
      date: '2024-01-13',
      time: '8:30 PM',
      image: '/api/placeholder/150/150',
      foods: ['Pasta Carbonara', 'Parmesan', 'Bacon', 'Eggs'],
      totalCalories: 650,
      protein: 25,
      carbs: 65,
      fats: 28,
      confidence: 93,
      model: 'expert_ensemble',
      sessionId: 'sess_004_20240113_2030'
    },
    {
      id: 5,
      date: '2024-01-13',
      time: '12:00 PM',
      image: '/api/placeholder/150/150',
      foods: ['Caesar Salad', 'Chicken Breast', 'Croutons', 'Dressing'],
      totalCalories: 380,
      protein: 32,
      carbs: 25,
      fats: 18,
      confidence: 89,
      model: 'expert_ensemble',
      sessionId: 'sess_005_20240113_1200'
    },
    {
      id: 6,
      date: '2024-01-12',
      time: '6:45 PM',
      image: '/api/placeholder/150/150',
      foods: ['Beef Stir Fry', 'Broccoli', 'Carrots', 'Soy Sauce'],
      totalCalories: 450,
      protein: 38,
      carbs: 35,
      fats: 20,
      confidence: 95,
      model: 'expert_ensemble',
      sessionId: 'sess_006_20240112_1845'
    }
  ]
};

const History: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedFilter, setSelectedFilter] = useState('all');
  const [sortBy, setSortBy] = useState('date');
  const [sortOrder, setSortOrder] = useState('desc');
  const [selectedAnalyses, setSelectedAnalyses] = useState<number[]>([]);
  const [selectedAnalysisDetail, setSelectedAnalysisDetail] = useState<any>(null);

  // Filter and sort analyses
  const filteredAnalyses = mockHistoryData.analyses
    .filter(analysis => {
      const matchesSearch = analysis.foods.some(food => 
        food.toLowerCase().includes(searchTerm.toLowerCase())
      ) || analysis.sessionId.toLowerCase().includes(searchTerm.toLowerCase());
      
      const matchesFilter = selectedFilter === 'all' || 
        (selectedFilter === 'today' && analysis.date === new Date().toISOString().split('T')[0]) ||
        (selectedFilter === 'week' && isWithinWeek(analysis.date)) ||
        (selectedFilter === 'month' && isWithinMonth(analysis.date));
      
      return matchesSearch && matchesFilter;
    })
    .sort((a, b) => {
      let comparison = 0;
      switch (sortBy) {
        case 'date':
          comparison = new Date(b.date).getTime() - new Date(a.date).getTime();
          break;
        case 'calories':
          comparison = b.totalCalories - a.totalCalories;
          break;
        case 'confidence':
          comparison = b.confidence - a.confidence;
          break;
        case 'time':
          comparison = new Date(`${b.date} ${b.time}`).getTime() - new Date(`${a.date} ${a.time}`).getTime();
          break;
      }
      return sortOrder === 'desc' ? comparison : -comparison;
    });

  // Helper functions for date filtering
  const isWithinWeek = (date: string) => {
    const analysisDate = new Date(date);
    const weekAgo = new Date();
    weekAgo.setDate(weekAgo.getDate() - 7);
    return analysisDate >= weekAgo;
  };

  const isWithinMonth = (date: string) => {
    const analysisDate = new Date(date);
    const monthAgo = new Date();
    monthAgo.setMonth(monthAgo.getMonth() - 1);
    return analysisDate >= monthAgo;
  };

  // Handle analysis selection
  const toggleAnalysisSelection = (id: number) => {
    setSelectedAnalyses(prev => 
      prev.includes(id) 
        ? prev.filter(item => item !== id)
        : [...prev, id]
    );
  };

  // Handle analysis detail view
  const handleViewAnalysis = (analysis: any) => {
    setSelectedAnalysisDetail(analysis);
  };

  // Handle bulk actions
  const handleBulkDelete = () => {
    if (selectedAnalyses.length > 0) {
      if (confirm(`Are you sure you want to delete ${selectedAnalyses.length} analysis(es)?`)) {
        // Here you would typically call an API to delete the analyses
        console.log('Deleting analyses:', selectedAnalyses);
        setSelectedAnalyses([]);
      }
    }
  };

  const handleBulkExport = () => {
    if (selectedAnalyses.length > 0) {
      // Here you would typically generate and download a report
      console.log('Exporting analyses:', selectedAnalyses);
    }
  };

  // Calculate statistics
  const totalCalories = filteredAnalyses.reduce((sum, analysis) => sum + analysis.totalCalories, 0);
  const averageCalories = filteredAnalyses.length > 0 ? Math.round(totalCalories / filteredAnalyses.length) : 0;
  const averageConfidence = filteredAnalyses.length > 0 
    ? Math.round(filteredAnalyses.reduce((sum, analysis) => sum + analysis.confidence, 0) / filteredAnalyses.length)
    : 0;

  return (
    <div className="history-page">
      <Navigation />
      <div className="history-content">
        <div className="history-header">
          <Link to="/dashboard" className="back-button">
            <ArrowLeft size={20} />
            Back to Dashboard
          </Link>
          <div className="header-content">
            <h1>Analysis History</h1>
            <p>Track your nutrition journey and view detailed analysis results</p>
          </div>
        </div>

        {/* Enhanced Stats */}
        <div className="history-stats">
          <div className="stat-card">
            <div className="stat-icon">
              <Camera size={24} />
            </div>
            <div className="stat-content">
              <h3>{mockHistoryData.totalAnalyses}</h3>
              <p>Total Analyses</p>
            </div>
          </div>
          
          <div className="stat-card">
            <div className="stat-icon">
              <Calendar size={24} />
            </div>
            <div className="stat-content">
              <h3>{mockHistoryData.thisWeek}</h3>
              <p>This Week</p>
            </div>
          </div>
          
          <div className="stat-card">
            <div className="stat-icon">
              <Zap size={24} />
            </div>
            <div className="stat-content">
              <h3>{totalCalories.toLocaleString()}</h3>
              <p>Total Calories</p>
            </div>
          </div>
          
          <div className="stat-card">
            <div className="stat-icon">
              <Target size={24} />
            </div>
            <div className="stat-content">
              <h3>{averageCalories}</h3>
              <p>Avg Calories</p>
            </div>
          </div>
        </div>

        {/* Enhanced Filters and Search */}
        <div className="filters-section">
          <div className="search-box">
            <Search size={20} />
            <input
              type="text"
              placeholder="Search by food items or session ID..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
          
          <div className="filter-controls">
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
            
            <div className="sort-controls">
              <select 
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value)}
                className="sort-select"
              >
                <option value="date">Sort by Date</option>
                <option value="calories">Sort by Calories</option>
                <option value="confidence">Sort by Confidence</option>
                <option value="time">Sort by Time</option>
              </select>
              
              <button 
                className="sort-order-btn"
                onClick={() => setSortOrder(sortOrder === 'desc' ? 'asc' : 'desc')}
              >
                {sortOrder === 'desc' ? '↓' : '↑'}
              </button>
            </div>
          </div>
        </div>

        {/* Bulk Actions */}
        {selectedAnalyses.length > 0 && (
          <div className="bulk-actions">
            <div className="bulk-info">
              <span>{selectedAnalyses.length} analysis(es) selected</span>
            </div>
            <div className="bulk-buttons">
              <button className="bulk-btn export" onClick={handleBulkExport}>
                <Download size={16} />
                Export
              </button>
              <button className="bulk-btn delete" onClick={handleBulkDelete}>
                <Trash2 size={16} />
                Delete
              </button>
            </div>
          </div>
        )}

        {/* Enhanced History List */}
        <div className="history-list">
          {filteredAnalyses.length > 0 ? (
            filteredAnalyses.map((analysis) => (
              <div key={analysis.id} className="history-item">
                <div className="item-checkbox">
                  <input
                    type="checkbox"
                    checked={selectedAnalyses.includes(analysis.id)}
                    onChange={() => toggleAnalysisSelection(analysis.id)}
                  />
                </div>
                
                <div className="item-image">
                  <div className="placeholder-image">
                    <Camera size={24} />
                  </div>
                </div>
                
                <div className="item-content">
                  <div className="item-header">
                    <h3>
                      {analysis.foods.slice(0, 2).join(', ')}
                      {analysis.foods.length > 2 && ` +${analysis.foods.length - 2} more`}
                    </h3>
                    <div className="item-meta">
                      <span className="item-date">
                        <Calendar size={14} />
                        {new Date(analysis.date).toLocaleDateString()}
                      </span>
                      <span className="item-time">
                        <Clock size={14} />
                        {analysis.time}
                      </span>
                      <span className="item-confidence">
                        <Target size={14} />
                        {analysis.confidence}% confidence
                      </span>
                    </div>
                  </div>
                  
                  <div className="item-details">
                    <div className="nutrition-breakdown">
                      <span className="nutrition-item">
                        <Zap size={14} />
                        {analysis.totalCalories} cal
                      </span>
                      <span className="nutrition-item">
                        <Apple size={14} />
                        P: {analysis.protein}g
                      </span>
                      <span className="nutrition-item">
                        <BarChart3 size={14} />
                        C: {analysis.carbs}g
                      </span>
                      <span className="nutrition-item">
                        <TrendingUp size={14} />
                        F: {analysis.fats}g
                      </span>
                    </div>
                    
                    <div className="food-tags">
                      {analysis.foods.slice(0, 3).map((food, index) => (
                        <span key={index} className="food-tag">{food}</span>
                      ))}
                      {analysis.foods.length > 3 && (
                        <span className="food-tag more">+{analysis.foods.length - 3} more</span>
                      )}
                    </div>
                  </div>
                  
                  <div className="item-footer">
                    <span className="session-id">ID: {analysis.sessionId}</span>
                    <span className="model-used">Model: {analysis.model}</span>
                  </div>
                </div>
                
                <div className="item-actions">
                  <button className="action-btn view" title="View Details" onClick={() => handleViewAnalysis(analysis)}>
                    <Eye size={16} />
                  </button>
                  <button className="action-btn export" title="Export Analysis">
                    <Download size={16} />
                  </button>
                  <button className="action-btn delete" title="Delete Analysis">
                    <Trash2 size={16} />
                  </button>
                </div>
              </div>
            ))
          ) : (
            <div className="empty-state">
              <div className="empty-icon">
                <Camera size={48} />
              </div>
              <h2>No analyses found</h2>
              <p>
                {searchTerm || selectedFilter !== 'all' 
                  ? 'Try adjusting your search or filters'
                  : 'Start by analyzing your first meal'
                }
              </p>
              {!searchTerm && selectedFilter === 'all' && (
                <Link to="/analysis" className="btn btn-primary">
                  <Camera size={20} />
                  Start Analysis
                </Link>
              )}
            </div>
          )}
        </div>

        {/* Selected Analysis Detail Panel */}
        {selectedAnalysisDetail && (
          <div className="selected-analysis">
            <h3>
              {selectedAnalysisDetail.foods.slice(0, 2).join(', ')}
              {selectedAnalysisDetail.foods.length > 2 && ` +${selectedAnalysisDetail.foods.length - 2} more`}
            </h3>
            
            <div className="analysis-meta">
              <div className="meta-item">
                <Calendar size={16} />
                {new Date(selectedAnalysisDetail.date).toLocaleDateString()}
              </div>
              <div className="meta-item">
                <Clock size={16} />
                {selectedAnalysisDetail.time}
              </div>
              <div className="meta-item">
                <Target size={16} />
                {selectedAnalysisDetail.confidence}% confidence
              </div>
            </div>
            
            <div className="nutrition-summary">
              <div className="nutrition-item">
                <Zap size={16} />
                {selectedAnalysisDetail.totalCalories} cal
              </div>
              <div className="nutrition-item">
                <Apple size={16} />
                P: {selectedAnalysisDetail.protein}g
              </div>
              <div className="nutrition-item">
                <BarChart3 size={16} />
                C: {selectedAnalysisDetail.carbs}g
              </div>
              <div className="nutrition-item">
                <TrendingUp size={16} />
                F: {selectedAnalysisDetail.fats}g
              </div>
            </div>
            
            <div className="food-tags">
              {selectedAnalysisDetail.foods.map((food: string, index: number) => (
                <span key={index} className="food-tag">{food}</span>
              ))}
            </div>
          </div>
        )}

        {/* Results Summary */}
        {filteredAnalyses.length > 0 && (
          <div className="results-summary">
            <div className="summary-item">
              <span className="summary-label">Showing</span>
              <span className="summary-value">{filteredAnalyses.length} of {mockHistoryData.analyses.length} analyses</span>
            </div>
            <div className="summary-item">
              <span className="summary-label">Total Calories</span>
              <span className="summary-value">{totalCalories.toLocaleString()} cal</span>
            </div>
            <div className="summary-item">
              <span className="summary-label">Average Confidence</span>
              <span className="summary-value">{averageConfidence}%</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default History;
