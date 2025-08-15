import React, { useState } from 'react';
import {
  Calendar,
  Coffee,
  Utensils,
  Apple,
  Droplets,
  ChevronDown,
  ChevronUp,
  Clock,
  Target
} from 'lucide-react';
import './DailyMealPlan.css';

interface DailyMealPlanProps {
  mealPlan: {
    breakfast: string[];
    lunch: string[];
    dinner: string[];
    snacks: string[];
    hydration: string[];
    totalCalories: number;
    notes: string;
  };
}

const DailyMealPlan: React.FC<DailyMealPlanProps> = ({ mealPlan }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const mealSections = [
    {
      title: 'Breakfast',
      icon: Coffee,
      items: mealPlan.breakfast,
      color: '#fbbf24',
      time: '7:00 AM - 9:00 AM'
    },
    {
      title: 'Lunch',
      icon: Utensils,
      items: mealPlan.lunch,
      color: '#22c55e',
      time: '12:00 PM - 2:00 PM'
    },
    {
      title: 'Dinner',
      icon: Utensils,
      items: mealPlan.dinner,
      color: '#3b82f6',
      time: '6:00 PM - 8:00 PM'
    },
    {
      title: 'Snacks',
      icon: Apple,
      items: mealPlan.snacks,
      color: '#8b5cf6',
      time: 'Throughout the day'
    },
    {
      title: 'Hydration',
      icon: Droplets,
      items: mealPlan.hydration,
      color: '#06b6d4',
      time: 'All day'
    }
  ];

  return (
    <div className="daily-meal-plan">
      <div className="meal-plan-header">
        <div className="meal-plan-title">
          <Calendar size={24} className="meal-plan-icon" />
          <h3>Daily Meal Plan</h3>
          <div className="meal-plan-badge">
            <Target size={16} />
            <span>{mealPlan.totalCalories} cal</span>
          </div>
        </div>
        <button
          className="expand-button"
          onClick={() => setIsExpanded(!isExpanded)}
        >
          {isExpanded ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
        </button>
      </div>

      <div className="meal-plan-summary">
        <div className="calories-overview">
          <div className="calories-circle">
            <span className="calories-number">{mealPlan.totalCalories}</span>
            <span className="calories-label">calories</span>
          </div>
          <div className="calories-info">
            <h4>Daily Target</h4>
            <p>Complete meal plan for the rest of your day</p>
          </div>
        </div>
      </div>

      {isExpanded && (
        <div className="meal-plan-details">
          <div className="meal-sections">
            {mealSections.map((section, index) => (
              <div key={index} className="meal-section">
                <div className="section-header">
                  <section.icon size={20} style={{ color: section.color }} />
                  <h4>{section.title}</h4>
                  <div className="meal-time">
                    <Clock size={14} />
                    <span>{section.time}</span>
                  </div>
                </div>
                <div className="meal-items">
                  {section.items.map((item, itemIndex) => (
                    <div key={itemIndex} className="meal-item">
                      <div className="item-bullet" style={{ backgroundColor: section.color }}></div>
                      <span>{item}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {mealPlan.notes && (
            <div className="meal-plan-notes">
              <h4>Notes</h4>
              <p>{mealPlan.notes}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default DailyMealPlan;
