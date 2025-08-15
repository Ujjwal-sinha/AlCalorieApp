import React, { useState } from 'react';
import './FoodItemReports.css';

interface FoodItemReport {
  nutritionProfile: string;
  healthBenefits: string;
  nutritionalHistory: string;
  cookingMethods: string;
  servingSuggestions: string;
  potentialConcerns: string;
  alternatives: string;
}

interface FoodItemReportsProps {
  foodItemReports: {
    [foodName: string]: FoodItemReport;
  };
}

const FoodItemReports: React.FC<FoodItemReportsProps> = ({ foodItemReports }) => {
  const [expandedItems, setExpandedItems] = useState<Set<string>>(new Set());

  const toggleItem = (foodName: string) => {
    const newExpanded = new Set(expandedItems);
    if (newExpanded.has(foodName)) {
      newExpanded.delete(foodName);
    } else {
      newExpanded.add(foodName);
    }
    setExpandedItems(newExpanded);
  };

  const getFoodIcon = (foodName: string) => {
    const name = foodName.toLowerCase();
    if (name.includes('apple') || name.includes('fruit')) return '🍎';
    if (name.includes('chicken') || name.includes('meat')) return '🍗';
    if (name.includes('salad') || name.includes('vegetable')) return '🥗';
    if (name.includes('bread') || name.includes('toast')) return '🍞';
    if (name.includes('rice') || name.includes('grain')) return '🍚';
    if (name.includes('fish') || name.includes('salmon')) return '🐟';
    if (name.includes('egg')) return '🥚';
    if (name.includes('milk') || name.includes('dairy')) return '🥛';
    if (name.includes('nut') || name.includes('almond')) return '🥜';
    return '🍽️';
  };

  return (
    <div className="food-item-reports">
      <div className="reports-header">
        <h3>📋 Detailed Food Analysis</h3>
        <p>Comprehensive nutrition profiles and health insights for each detected food item</p>
      </div>

      <div className="reports-grid">
        {Object.entries(foodItemReports).map(([foodName, report]) => (
          <div key={foodName} className="food-report-card">
            <div 
              className="report-header"
              onClick={() => toggleItem(foodName)}
            >
              <div className="food-info">
                <span className="food-icon">{getFoodIcon(foodName)}</span>
                <h4>{foodName}</h4>
              </div>
              <span className={`expand-icon ${expandedItems.has(foodName) ? 'expanded' : ''}`}>
                ▼
              </span>
            </div>

            {expandedItems.has(foodName) && (
              <div className="report-content">
                <div className="report-section">
                  <h5>🥗 Nutrition Profile</h5>
                  <p>{report.nutritionProfile}</p>
                </div>

                <div className="report-section">
                  <h5>💪 Health Benefits</h5>
                  <p>{report.healthBenefits}</p>
                </div>

                <div className="report-section">
                  <h5>📚 Nutritional History</h5>
                  <p>{report.nutritionalHistory}</p>
                </div>

                <div className="report-section">
                  <h5>👨‍🍳 Cooking Methods</h5>
                  <p>{report.cookingMethods}</p>
                </div>

                <div className="report-section">
                  <h5>🍽️ Serving Suggestions</h5>
                  <p>{report.servingSuggestions}</p>
                </div>

                <div className="report-section">
                  <h5>⚠️ Potential Concerns</h5>
                  <p>{report.potentialConcerns}</p>
                </div>

                <div className="report-section">
                  <h5>🔄 Alternatives</h5>
                  <p>{report.alternatives}</p>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default FoodItemReports;
