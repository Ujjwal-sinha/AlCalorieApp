import React, { useState, useEffect } from 'react';
import { 
  Camera, 
  BarChart3, 
  Calendar,
  Target,
 
  Zap,
  ArrowUp,
  ArrowDown,
  Eye,
  Download,
  TrendingUp,
  TrendingDown,
  Clock,

  Award,
  Sparkles,

  Brain,
  
  ChevronRight,
  Star,
  CheckCircle,
  AlertCircle,
  Info
} from 'lucide-react';
import { Link } from 'react-router-dom';
import Navigation from '../components/Navigation';
import './Dashboard.css';

// Enhanced mock data for charts and analytics
const mockAnalyticsData = {
  totalAnalyses: 156,
  thisWeek: 23,
  thisMonth: 89,
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
    { name: 'Chicken Breast', calories: 165, frequency: 12, trend: 'up' },
    { name: 'Brown Rice', calories: 216, frequency: 10, trend: 'stable' },
    { name: 'Broccoli', calories: 55, frequency: 8, trend: 'up' },
    { name: 'Salmon', calories: 208, frequency: 7, trend: 'down' },
    { name: 'Sweet Potato', calories: 103, frequency: 6, trend: 'up' }
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
      fats: 18,
      accuracy: 98.5,
      status: 'completed'
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
      fats: 22,
      accuracy: 97.2,
      status: 'completed'
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
      fats: 15,
      accuracy: 99.1,
      status: 'completed'
    }
  ],
  achievements: [
    { id: 1, title: 'First Analysis', description: 'Completed your first food analysis', icon: '🎯', unlocked: true },
    { id: 2, title: 'Week Warrior', description: 'Analyzed food for 7 consecutive days', icon: '🔥', unlocked: true },
    { id: 3, title: 'Accuracy Master', description: 'Achieved 99%+ accuracy 10 times', icon: '⭐', unlocked: false },
    { id: 4, title: 'Variety Explorer', description: 'Analyzed 50 different foods', icon: '🌍', unlocked: false }
  ],
  insights: [
    {
      id: 1,
      type: 'positive',
      title: 'Great Protein Intake',
      description: 'Your protein consumption is 15% above your daily goal',
      icon: <TrendingUp size={20} />
    },
    {
      id: 2,
      type: 'warning',
      title: 'Low Fiber Alert',
      description: 'Consider adding more fiber-rich foods to your diet',
      icon: <AlertCircle size={20} />
    },
    {
      id: 3,
      type: 'info',
      title: 'Calorie Balance',
      description: 'You\'re maintaining a healthy calorie balance this week',
      icon: <Info size={20} />
    }
  ]
};

const Dashboard: React.FC = () => {
  const [selectedPeriod, setSelectedPeriod] = useState('week');
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    // Simulate loading state
    setIsLoading(true);
    const timer = setTimeout(() => setIsLoading(false), 1000);
    return () => clearTimeout(timer);
  }, []);

  // Calculate enhanced metrics
  const totalCalories = mockAnalyticsData.recentAnalyses.reduce((sum, analysis) => sum + analysis.totalCalories, 0);
  const averageCaloriesPerMeal = Math.round(totalCalories / mockAnalyticsData.recentAnalyses.length);
  const totalAccuracy = mockAnalyticsData.recentAnalyses.reduce((sum, analysis) => sum + analysis.accuracy, 0);
  const averageAccuracy = Math.round(totalAccuracy / mockAnalyticsData.recentAnalyses.length);

  // Calculate nutrition totals

  const renderEnhancedStats = () => {
    const stats = [
      {
        icon: <Camera size={24} />,
        number: mockAnalyticsData.totalAnalyses,
        label: 'Total Analyses',
        trend: '+12%',
        trendDirection: 'up',
        color: '#4caf50',
        subtitle: 'Lifetime total'
      },
      {
        icon: <Calendar size={24} />,
        number: mockAnalyticsData.thisWeek,
        label: 'This Week',
        trend: '+5',
        trendDirection: 'up',
        color: '#2196f3',
        subtitle: 'vs last week'
      },
      {
        icon: <Zap size={24} />,
        number: totalCalories,
        label: 'Total Calories',
        trend: averageCaloriesPerMeal,
        trendDirection: 'stable',
        color: '#ff9800',
        subtitle: 'Last 3 meals'
      },
      {
        icon: <Target size={24} />,
        number: mockAnalyticsData.goalProgress,
        label: 'Goal Progress',
        trend: averageAccuracy,
        trendDirection: 'up',
        color: '#9c27b0',
        subtitle: '% accuracy'
      }
    ];

    return (
      <div className="stats-grid">
        {stats.map((stat, index) => (
          <div key={index} className="stat-card" style={{ '--stat-color': stat.color } as React.CSSProperties}>
            <div className="stat-header">
              <div className="stat-icon">
                {stat.icon}
              </div>
              <div className="stat-trend">
                {stat.trendDirection === 'up' ? <ArrowUp size={16} /> : <ArrowDown size={16} />}
                <span>{stat.trend}</span>
              </div>
            </div>
            <div className="stat-number">{stat.number}</div>
            <div className="stat-label">{stat.label}</div>
            <div className="stat-subtitle">{stat.subtitle}</div>
            {index === 3 && (
              <div className="progress-ring">
                <svg width="60" height="60">
                  <circle
                    cx="30"
                    cy="30"
                    r="25"
                    fill="none"
                    stroke="rgba(156, 39, 176, 0.2)"
                    strokeWidth="4"
                  />
                  <circle
                    cx="30"
                    cy="30"
                    r="25"
                    fill="none"
                    stroke="#9c27b0"
                    strokeWidth="4"
                    strokeDasharray={`${(mockAnalyticsData.goalProgress / 100) * 157} 157`}
                    strokeDashoffset="0"
                    transform="rotate(-90 30 30)"
                  />
                </svg>
              </div>
            )}
          </div>
        ))}
      </div>
    );
  };

  const renderQuickActions = () => (
    <div className="quick-actions">
      <div className="section-header">
        <div className="section-badge">
          <Zap size={16} />
          <span>Quick Actions</span>
        </div>
        <h2>What would you like to do?</h2>
        <p>Get started with your nutrition tracking journey</p>
      </div>
      <div className="actions-grid">
        <Link to="/analysis" className="action-card primary">
          <div className="action-icon">
            <Camera size={24} />
          </div>
          <div className="action-content">
            <h3>New Analysis</h3>
            <p>Analyze a new meal with AI</p>
          </div>
          <ChevronRight size={20} />
        </Link>
        
        <Link to="/history" className="action-card">
          <div className="action-icon">
            <BarChart3 size={24} />
          </div>
          <div className="action-content">
            <h3>View History</h3>
            <p>See your past analyses</p>
          </div>
          <ChevronRight size={20} />
        </Link>
        
        <Link to="/settings" className="action-card">
          <div className="action-icon">
            <Target size={24} />
          </div>
          <div className="action-content">
            <h3>Set Goals</h3>
            <p>Update nutrition goals</p>
          </div>
          <ChevronRight size={20} />
        </Link>
        
        <div className="action-card">
          <div className="action-icon">
            <Download size={24} />
          </div>
          <div className="action-content">
            <h3>Export Data</h3>
            <p>Download your reports</p>
          </div>
          <ChevronRight size={20} />
        </div>
      </div>
    </div>
  );

  const renderInsights = () => (
    <div className="insights-section">
      <div className="section-header">
        <div className="section-badge">
          <Brain size={16} />
          <span>AI Insights</span>
        </div>
        <h2>Smart Recommendations</h2>
        <p>Personalized insights based on your nutrition data</p>
      </div>
      <div className="insights-grid">
        {mockAnalyticsData.insights.map((insight) => (
          <div key={insight.id} className={`insight-card ${insight.type}`}>
            <div className="insight-icon">
              {insight.icon}
            </div>
            <div className="insight-content">
              <h3>{insight.title}</h3>
              <p>{insight.description}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const renderAchievements = () => (
    <div className="achievements-section">
      <div className="section-header">
        <div className="section-badge">
          <Award size={16} />
          <span>Achievements</span>
        </div>
        <h2>Your Progress</h2>
        <p>Track your milestones and accomplishments</p>
      </div>
      <div className="achievements-grid">
        {mockAnalyticsData.achievements.map((achievement) => (
          <div key={achievement.id} className={`achievement-card ${achievement.unlocked ? 'unlocked' : 'locked'}`}>
            <div className="achievement-icon">
              <span>{achievement.icon}</span>
            </div>
            <div className="achievement-content">
              <h3>{achievement.title}</h3>
              <p>{achievement.description}</p>
            </div>
            {achievement.unlocked && (
              <div className="achievement-badge">
                <CheckCircle size={16} />
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );

  const renderEnhancedAnalytics = () => (
    <div className="analytics-section">
      <div className="section-header">
        <div className="section-badge">
          <BarChart3 size={16} />
          <span>Analytics</span>
        </div>
        <h2>Your Nutrition Analytics</h2>
        <p>Comprehensive insights into your eating patterns</p>
      </div>
      <div className="analytics-grid">
        {renderCalorieChart()}
        {renderNutritionChart()}
        {renderTopFoodsChart()}
      </div>
    </div>
  );

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
              <div className="food-meta">
                <span className="food-frequency">{food.frequency} times</span>
                <div className={`food-trend ${food.trend}`}>
                  {food.trend === 'up' ? <TrendingUp size={12} /> : 
                   food.trend === 'down' ? <TrendingDown size={12} /> : 
                   <span>—</span>}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderRecentAnalyses = () => (
    <div className="recent-analyses">
      <div className="section-header">
        <div className="section-badge">
          <Clock size={16} />
          <span>Recent</span>
        </div>
        <h2>Recent Analyses</h2>
        <p>Your latest nutrition insights</p>
      </div>
      <div className="analyses-list">
        {mockAnalyticsData.recentAnalyses.map((analysis) => (
          <div key={analysis.id} className="analysis-item">
            <div className="analysis-icon">
              <Camera size={20} />
            </div>
            <div className="analysis-content">
              <div className="analysis-header">
                <div className="analysis-title">
                  {analysis.foods.slice(0, 2).join(', ')}
                  {analysis.foods.length > 2 && ` +${analysis.foods.length - 2} more`}
                </div>
                <div className="analysis-accuracy">
                  <Star size={12} />
                  <span>{analysis.accuracy}%</span>
                </div>
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
  );

  if (isLoading) {
    return (
      <div className="dashboard-page">
        <Navigation />
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Loading your dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard-page">
      <Navigation />
      <div className="dashboard-content">
        <div className="dashboard-header">
          <div className="welcome-section">
            <div className="welcome-badge">
              <Sparkles size={16} />
              <span>Welcome Back</span>
            </div>
            <h1 className="welcome-title">
              Good morning! 👋
            </h1>
            <p className="welcome-subtitle">
              Here's your nutrition overview for today
            </p>
          </div>
        </div>

        {renderEnhancedStats()}
        {renderQuickActions()}
        {renderInsights()}
        {renderAchievements()}
        {renderEnhancedAnalytics()}
        {renderRecentAnalyses()}
      </div>
    </div>
  );
};

export default Dashboard;
