import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { TrendingUp, Calendar, Target, Zap } from 'lucide-react';
import { HistoryService } from '../services/HistoryService';
import type { TrendData } from '../types';

const TrendsView: React.FC = () => {
  const [trendData, setTrendData] = useState<TrendData[]>([]);
  const [statistics, setStatistics] = useState({
    totalAnalyses: 0,
    averageCalories: 0,
    mostCommonFood: 'N/A',
    totalCalories: 0
  });
  const [selectedPeriod, setSelectedPeriod] = useState<7 | 30 | 90>(30);

  useEffect(() => {
    const historyService = HistoryService.getInstance();
    const trends = historyService.generateTrendData(selectedPeriod);
    const stats = historyService.getStatistics();
    
    setTrendData(trends);
    setStatistics(stats);
  }, [selectedPeriod]);

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  };

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="chart-tooltip">
          <p className="tooltip-label">{formatDate(label)}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} style={{ color: entry.color }}>
              {entry.name}: {entry.value}
              {entry.name === 'Calories' ? ' cal' : 'g'}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  const hasData = trendData.some(day => day.calories > 0);

  return (
    <div className="trends-view">
      <div className="trends-header">
        <h2>
          <TrendingUp size={24} />
          Nutritional Trends
        </h2>
        <p>Track your nutritional intake over time</p>
      </div>

      {/* Period Selection */}
      <div className="period-selector">
        <button
          className={`period-button ${selectedPeriod === 7 ? 'active' : ''}`}
          onClick={() => setSelectedPeriod(7)}
        >
          7 Days
        </button>
        <button
          className={`period-button ${selectedPeriod === 30 ? 'active' : ''}`}
          onClick={() => setSelectedPeriod(30)}
        >
          30 Days
        </button>
        <button
          className={`period-button ${selectedPeriod === 90 ? 'active' : ''}`}
          onClick={() => setSelectedPeriod(90)}
        >
          90 Days
        </button>
      </div>

      {/* Statistics Cards */}
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-icon">
            <Target size={24} />
          </div>
          <div className="stat-content">
            <span className="stat-value">{statistics.totalAnalyses}</span>
            <span className="stat-label">Total Analyses</span>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">
            <Zap size={24} />
          </div>
          <div className="stat-content">
            <span className="stat-value">{statistics.averageCalories}</span>
            <span className="stat-label">Avg Calories/Meal</span>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">
            <Calendar size={24} />
          </div>
          <div className="stat-content">
            <span className="stat-value">{statistics.totalCalories.toLocaleString()}</span>
            <span className="stat-label">Total Calories</span>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">
            <TrendingUp size={24} />
          </div>
          <div className="stat-content">
            <span className="stat-value">{statistics.mostCommonFood}</span>
            <span className="stat-label">Most Common Food</span>
          </div>
        </div>
      </div>

      {hasData ? (
        <div className="charts-container">
          {/* Calorie Trends */}
          <div className="chart-card">
            <h3>Daily Calorie Intake</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={trendData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="date" 
                  tickFormatter={formatDate}
                  interval="preserveStartEnd"
                />
                <YAxis />
                <Tooltip content={<CustomTooltip />} />
                <Line 
                  type="monotone" 
                  dataKey="calories" 
                  stroke="#FF6B6B" 
                  strokeWidth={2}
                  dot={{ fill: '#FF6B6B', strokeWidth: 2, r: 4 }}
                  name="Calories"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Macronutrient Trends */}
          <div className="chart-card">
            <h3>Macronutrient Trends</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={trendData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="date" 
                  tickFormatter={formatDate}
                  interval="preserveStartEnd"
                />
                <YAxis />
                <Tooltip content={<CustomTooltip />} />
                <Line 
                  type="monotone" 
                  dataKey="protein" 
                  stroke="#4ECDC4" 
                  strokeWidth={2}
                  name="Protein"
                />
                <Line 
                  type="monotone" 
                  dataKey="carbs" 
                  stroke="#45B7D1" 
                  strokeWidth={2}
                  name="Carbs"
                />
                <Line 
                  type="monotone" 
                  dataKey="fats" 
                  stroke="#F9CA24" 
                  strokeWidth={2}
                  name="Fats"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Weekly Summary */}
          <div className="chart-card">
            <h3>Weekly Calorie Summary</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={trendData.filter((_, index) => index % 7 === 0)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="date" 
                  tickFormatter={formatDate}
                />
                <YAxis />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="calories" fill="#667eea" name="Calories" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      ) : (
        <div className="no-data">
          <Calendar size={64} className="no-data-icon" />
          <h3>No Data Available</h3>
          <p>Start analyzing your meals to see trends and statistics here.</p>
        </div>
      )}
    </div>
  );
};

export default TrendsView;