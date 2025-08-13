import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { 
  Camera, 
  BarChart3, 
  TrendingUp, 
  Target, 
  Calendar,
  ArrowRight,
  Plus,
  Activity,
  Zap,
  Award
} from 'lucide-react';
import Navigation from '../components/Navigation';
import './Dashboard.css';

const Dashboard: React.FC = () => {
  const [stats, setStats] = useState({
    totalAnalyses: 0,
    thisWeek: 0,
    averageCalories: 0,
    goalProgress: 0
  });

  useEffect(() => {
    // Simulate loading stats
    const timer = setTimeout(() => {
      setStats({
        totalAnalyses: 47,
        thisWeek: 12,
        averageCalories: 1850,
        goalProgress: 78
      });
    }, 1000);

    return () => clearTimeout(timer);
  }, []);

  const quickActions = [
    {
      title: 'Analyze Food',
      description: 'Take a photo or upload an image',
      icon: <Camera size={24} />,
      link: '/analysis',
      color: 'primary'
    },
    {
      title: 'View History',
      description: 'Check your previous analyses',
      icon: <BarChart3 size={24} />,
      link: '/history',
      color: 'secondary'
    },
    {
      title: 'Set Goals',
      description: 'Configure your nutrition targets',
      icon: <Target size={24} />,
      link: '/settings',
      color: 'accent'
    }
  ];

  const recentAnalyses = [
    {
      id: 1,
      food: 'Grilled Chicken Salad',
      calories: 320,
      time: '2 hours ago',
      image: '🍗'
    },
    {
      id: 2,
      food: 'Pasta Carbonara',
      calories: 650,
      time: '1 day ago',
      image: '🍝'
    },
    {
      id: 3,
      food: 'Greek Yogurt Bowl',
      calories: 180,
      time: '2 days ago',
      image: '🥣'
    }
  ];

  return (
    <div className="dashboard">
      <Navigation />
      
      <div className="dashboard-content">
        <div className="dashboard-header">
          <div className="header-content">
            <h1>Welcome back!</h1>
            <p>Track your nutrition and stay healthy with AI-powered analysis</p>
          </div>
          <div className="header-actions">
            <Link to="/analysis" className="btn btn-primary">
              <Camera size={20} />
              New Analysis
              <ArrowRight size={20} />
            </Link>
          </div>
        </div>

        {/* Stats Grid */}
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-icon">
              <Activity size={24} />
            </div>
            <div className="stat-content">
              <h3>{stats.totalAnalyses}</h3>
              <p>Total Analyses</p>
            </div>
          </div>

          <div className="stat-card">
            <div className="stat-icon">
              <Calendar size={24} />
            </div>
            <div className="stat-content">
              <h3>{stats.thisWeek}</h3>
              <p>This Week</p>
            </div>
          </div>

          <div className="stat-card">
            <div className="stat-icon">
              <Zap size={24} />
            </div>
            <div className="stat-content">
              <h3>{stats.averageCalories}</h3>
              <p>Avg Calories</p>
            </div>
          </div>

          <div className="stat-card">
            <div className="stat-icon">
              <Award size={24} />
            </div>
            <div className="stat-content">
              <h3>{stats.goalProgress}%</h3>
              <p>Goal Progress</p>
            </div>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="section">
          <div className="section-header">
            <h2>Quick Actions</h2>
            <p>Get started with your nutrition tracking</p>
          </div>
          <div className="quick-actions">
            {quickActions.map((action, index) => (
              <Link key={index} to={action.link} className={`action-card ${action.color}`}>
                <div className="action-icon">
                  {action.icon}
                </div>
                <div className="action-content">
                  <h3>{action.title}</h3>
                  <p>{action.description}</p>
                </div>
                <ArrowRight size={20} className="action-arrow" />
              </Link>
            ))}
          </div>
        </div>

        {/* Recent Analyses */}
        <div className="section">
          <div className="section-header">
            <h2>Recent Analyses</h2>
            <Link to="/history" className="view-all">
              View All
              <ArrowRight size={16} />
            </Link>
          </div>
          <div className="recent-analyses">
            {recentAnalyses.map((analysis) => (
              <div key={analysis.id} className="analysis-card">
                <div className="analysis-icon">
                  <span>{analysis.image}</span>
                </div>
                <div className="analysis-content">
                  <h3>{analysis.food}</h3>
                  <p>{analysis.calories} calories</p>
                  <span className="analysis-time">{analysis.time}</span>
                </div>
                <div className="analysis-actions">
                  <button className="btn-icon">
                    <Plus size={16} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Nutrition Insights */}
        <div className="section">
          <div className="section-header">
            <h2>Nutrition Insights</h2>
            <p>Your weekly nutrition summary</p>
          </div>
          <div className="insights-grid">
            <div className="insight-card">
              <div className="insight-header">
                <TrendingUp size={20} />
                <h3>Calorie Trend</h3>
              </div>
              <div className="insight-content">
                <p>You're averaging 1,850 calories per day this week, which is within your target range.</p>
                <div className="trend-indicator positive">
                  <span>+5%</span>
                  <span>vs last week</span>
                </div>
              </div>
            </div>

            <div className="insight-card">
              <div className="insight-header">
                <Target size={20} />
                <h3>Goal Progress</h3>
              </div>
              <div className="insight-content">
                <p>You're 78% towards your weekly nutrition goal. Keep up the great work!</p>
                <div className="progress-bar">
                  <div className="progress-fill" style={{ width: '78%' }}></div>
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
