import React, { useState } from 'react';
import { 
  Camera, 
  BarChart3, 
  Calendar,
  Target,
  Activity,
  Apple,
  Zap,
  ArrowUp,
  Eye,
  Download
} from 'lucide-react';
import { Link } from 'react-router-dom';
import Navigation from '../components/Navigation';
import './Dashboard.css';

// Mock data for charts and analytics
const mockAnalyticsData = {
  totalAnalyses: 156,
  thisWeek: 23,
  averageCalories: 1850,
  goalProgress: 78,
  weeklyCalories: [2100, 1950, 2200, 1800, 2000, 1750, 1900],
  monthlyTrend: [1850, 1920, 1880, 1950, 1820, 1980, 2050, 1900, 1850, 1920, 1880, 1950],
  nutritionBreakdown: {
    protein: 25,
    carbs: 45,
    fats: 30
  },
  topFoods: [
    { name: 'Chicken Breast', calories: 165, frequency: 12 },
    { name: 'Brown Rice', calories: 216, frequency: 10 },
    { name: 'Broccoli', calories: 55, frequency: 8 },
    { name: 'Salmon', calories: 208, frequency: 7 },
    { name: 'Sweet Potato', calories: 103, frequency: 6 }
  ],
  recentAnalyses: [
    {
      id: 1,
      date: '2024-01-15',
      time: '12:30 PM',
      image: '/api/placeholder/150/150',
      foods: ['Grilled Chicken', 'Quinoa', 'Mixed Vegetables'],
      totalCalories: 485,
      protein: 35,
      carbs: 45,
      fats: 18
    },
    {
      id: 2,
      date: '2024-01-14',
      time: '7:45 PM',
      image: '/api/placeholder/150/150',
      foods: ['Salmon', 'Brown Rice', 'Asparagus'],
      totalCalories: 520,
      protein: 42,
      carbs: 38,
      fats: 22
    },
    {
      id: 3,
      date: '2024-01-14',
      time: '1:15 PM',
      image: '/api/placeholder/150/150',
      foods: ['Turkey Sandwich', 'Apple', 'Greek Yogurt'],
      totalCalories: 420,
      protein: 28,
      carbs: 52,
      fats: 15
    }
  ]
};

const Dashboard: React.FC = () => {
  const [selectedPeriod, setSelectedPeriod] = useState('week');

  // Calculate total calories from recent analyses
  const totalCalories = mockAnalyticsData.recentAnalyses.reduce((sum, analysis) => sum + analysis.totalCalories, 0);
  const averageCaloriesPerMeal = Math.round(totalCalories / mockAnalyticsData.recentAnalyses.length);

  // Calculate nutrition totals
  const totalNutrition = mockAnalyticsData.recentAnalyses.reduce((acc, analysis) => ({
    protein: acc.protein + analysis.protein,
    carbs: acc.carbs + analysis.carbs,
    fats: acc.fats + analysis.fats
  }), { protein: 0, carbs: 0, fats: 0 });

  const renderCalorieChart = () => {
    const data = selectedPeriod === 'week' ? mockAnalyticsData.weeklyCalories : mockAnalyticsData.monthlyTrend;
    const labels = selectedPeriod === 'week' 
      ? ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
      : ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    
    const maxValue = Math.max(...data);
    const minValue = Math.min(...data);

    return (
      <div className="chart-container">
        <div className="chart-header">
          <h3>Calorie Intake Trend</h3>
          <div className="chart-controls">
            <button 
              className={`period-btn ${selectedPeriod === 'week' ? 'active' : ''}`}
              onClick={() => setSelectedPeriod('week')}
            >
              Week
            </button>
            <button 
              className={`period-btn ${selectedPeriod === 'month' ? 'active' : ''}`}
              onClick={() => setSelectedPeriod('month')}
            >
              Month
            </button>
          </div>
        </div>
        <div className="chart-content">
          <div className="chart-bars">
            {data.map((value, index) => {
              const height = ((value - minValue) / (maxValue - minValue)) * 100;
              const isToday = selectedPeriod === 'week' && index === new Date().getDay() - 1;
              return (
                <div key={index} className="chart-bar-container">
                  <div 
                    className={`chart-bar ${isToday ? 'today' : ''}`}
                    style={{ height: `${height}%` }}
                  >
                    <span className="bar-value">{value}</span>
                  </div>
                  <span className="bar-label">{labels[index]}</span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    );
  };

  const renderNutritionChart = () => {
    const { protein, carbs, fats } = mockAnalyticsData.nutritionBreakdown;
    const total = protein + carbs + fats;

    return (
      <div className="chart-container">
        <h3>Nutrition Breakdown</h3>
        <div className="nutrition-chart">
          <div className="pie-chart">
            <div className="pie-segment protein" style={{ 
              transform: `rotate(${(protein / total) * 360}deg)`,
              background: `conic-gradient(#4caf50 0deg ${(protein / total) * 360}deg, transparent ${(protein / total) * 360}deg)`
            }}></div>
            <div className="pie-segment carbs" style={{ 
              transform: `rotate(${(protein / total) * 360 + (carbs / total) * 360}deg)`,
              background: `conic-gradient(#66bb6a 0deg ${(carbs / total) * 360}deg, transparent ${(carbs / total) * 360}deg)`
            }}></div>
            <div className="pie-segment fats" style={{ 
              transform: `rotate(${(protein / total) * 360 + (carbs / total) * 360 + (fats / total) * 360}deg)`,
              background: `conic-gradient(#81c784 0deg ${(fats / total) * 360}deg, transparent ${(fats / total) * 360}deg)`
            }}></div>
            <div className="pie-center">
              <span className="total-percentage">100%</span>
            </div>
          </div>
          <div className="nutrition-legend">
            <div className="legend-item">
              <div className="legend-color protein"></div>
              <span>Protein: {protein}%</span>
            </div>
            <div className="legend-item">
              <div className="legend-color carbs"></div>
              <span>Carbs: {carbs}%</span>
            </div>
            <div className="legend-item">
              <div className="legend-color fats"></div>
              <span>Fats: {fats}%</span>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderTopFoodsChart = () => {
    return (
      <div className="chart-container">
        <h3>Most Analyzed Foods</h3>
        <div className="foods-chart">
          {mockAnalyticsData.topFoods.map((food, index) => (
            <div key={index} className="food-item">
              <div className="food-info">
                <span className="food-name">{food.name}</span>
                <span className="food-calories">{food.calories} cal</span>
              </div>
              <div className="food-bar">
                <div 
                  className="food-bar-fill"
                  style={{ width: `${(food.frequency / 12) * 100}%` }}
                ></div>
              </div>
              <span className="food-frequency">{food.frequency} times</span>
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="dashboard-page">
      <Navigation />
      <div className="dashboard-content">
        <div className="dashboard-header">
          <div className="welcome-text">
            Welcome back! 👋
          </div>
          <div className="welcome-subtitle">
            Here's your nutrition overview for today
          </div>
        </div>

        {/* Enhanced Stats Grid */}
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-icon">
              <Camera size={24} />
            </div>
            <div className="stat-number">{mockAnalyticsData.totalAnalyses}</div>
            <div className="stat-label">Total Analyses</div>
            <div className="stat-trend positive">
              <ArrowUp size={16} />
              <span>+12% this week</span>
            </div>
          </div>
          
          <div className="stat-card">
            <div className="stat-icon">
              <Calendar size={24} />
            </div>
            <div className="stat-number">{mockAnalyticsData.thisWeek}</div>
            <div className="stat-label">This Week</div>
            <div className="stat-trend positive">
              <ArrowUp size={16} />
              <span>+5 from last week</span>
            </div>
          </div>
          
          <div className="stat-card highlight">
            <div className="stat-icon">
              <Zap size={24} />
            </div>
            <div className="stat-number">{totalCalories}</div>
            <div className="stat-label">Total Calories</div>
            <div className="stat-subtitle">Last 3 meals</div>
          </div>
          
          <div className="stat-card">
            <div className="stat-icon">
              <Target size={24} />
            </div>
            <div className="stat-number">{mockAnalyticsData.goalProgress}%</div>
            <div className="stat-label">Goal Progress</div>
            <div className="progress-ring">
              <svg width="60" height="60">
                <circle
                  cx="30"
                  cy="30"
                  r="25"
                  fill="none"
                  stroke="rgba(76, 175, 80, 0.2)"
                  strokeWidth="4"
                />
                <circle
                  cx="30"
                  cy="30"
                  r="25"
                  fill="none"
                  stroke="#4caf50"
                  strokeWidth="4"
                  strokeDasharray={`${(mockAnalyticsData.goalProgress / 100) * 157} 157`}
                  strokeDashoffset="0"
                  transform="rotate(-90 30 30)"
                />
              </svg>
            </div>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="quick-actions">
          <div className="section-title">Quick Actions</div>
          <div className="actions-grid">
            <Link to="/analysis" className="action-card">
              <div className="action-icon">
                <Camera size={24} />
              </div>
              <div className="action-title">New Analysis</div>
              <div className="action-description">Analyze a new meal</div>
            </Link>
            
            <Link to="/history" className="action-card">
              <div className="action-icon">
                <BarChart3 size={24} />
              </div>
              <div className="action-title">View History</div>
              <div className="action-description">See past analyses</div>
            </Link>
            
            <Link to="/settings" className="action-card">
              <div className="action-icon">
                <Target size={24} />
              </div>
              <div className="action-title">Set Goals</div>
              <div className="action-description">Update nutrition goals</div>
            </Link>
            
            <div className="action-card">
              <div className="action-icon">
                <Download size={24} />
              </div>
              <div className="action-title">Export Data</div>
              <div className="action-description">Download your reports</div>
            </div>
          </div>
        </div>

        {/* Charts Section */}
        <div className="charts-section">
          <div className="section-title">Analytics & Insights</div>
          <div className="charts-grid">
            {renderCalorieChart()}
            {renderNutritionChart()}
            {renderTopFoodsChart()}
          </div>
        </div>

        {/* Enhanced Recent Analyses */}
        <div className="recent-analyses">
          <div className="section-title">Recent Analyses</div>
          <div className="analyses-list">
            {mockAnalyticsData.recentAnalyses.map((analysis) => (
              <div key={analysis.id} className="analysis-item">
                <div className="analysis-icon">
                  <Camera size={20} />
                </div>
                <div className="analysis-content">
                  <div className="analysis-title">
                    {analysis.foods.slice(0, 2).join(', ')}
                    {analysis.foods.length > 2 && ` +${analysis.foods.length - 2} more`}
                  </div>
                  <div className="analysis-details">
                    <span className="analysis-calories">{analysis.totalCalories} calories</span>
                    <span className="analysis-time">{analysis.date} at {analysis.time}</span>
                  </div>
                  <div className="analysis-nutrition">
                    <span className="nutrition-item">P: {analysis.protein}g</span>
                    <span className="nutrition-item">C: {analysis.carbs}g</span>
                    <span className="nutrition-item">F: {analysis.fats}g</span>
                  </div>
                </div>
                <div className="analysis-actions">
                  <button className="action-btn">
                    <Eye size={16} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Nutrition Insights */}
        <div className="nutrition-insights">
          <div className="insight-card">
            <div className="insight-header">
              <div className="insight-icon">
                <Apple size={20} />
              </div>
              <div className="insight-title">Average Calories</div>
            </div>
            <div className="insight-content">
              <div className="insight-value">{averageCaloriesPerMeal} cal/meal</div>
              <div className="insight-description">
                Your average calorie intake per meal is well within recommended ranges.
              </div>
            </div>
          </div>
          
          <div className="insight-card">
            <div className="insight-header">
              <div className="insight-icon">
                <Activity size={20} />
              </div>
              <div className="insight-title">Nutrition Balance</div>
            </div>
            <div className="insight-content">
              <div className="nutrition-summary">
                <div className="nutrition-item">
                  <span className="nutrition-label">Protein</span>
                  <span className="nutrition-value">{Math.round(totalNutrition.protein / mockAnalyticsData.recentAnalyses.length)}g avg</span>
                </div>
                <div className="nutrition-item">
                  <span className="nutrition-label">Carbs</span>
                  <span className="nutrition-value">{Math.round(totalNutrition.carbs / mockAnalyticsData.recentAnalyses.length)}g avg</span>
                </div>
                <div className="nutrition-item">
                  <span className="nutrition-label">Fats</span>
                  <span className="nutrition-value">{Math.round(totalNutrition.fats / mockAnalyticsData.recentAnalyses.length)}g avg</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
