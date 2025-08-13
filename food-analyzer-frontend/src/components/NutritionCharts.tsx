import React from 'react';
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import type { NutritionalData } from '../types';

interface NutritionChartsProps {
  data: NutritionalData;
}

const NutritionCharts: React.FC<NutritionChartsProps> = ({ data }) => {
  const totalCalories = data.total_calories;
  
  // Macronutrient distribution data
  const macroData = [
    {
      name: 'Protein',
      value: data.total_protein * 4,
      percentage: Math.round((data.total_protein * 4 / totalCalories) * 100),
      color: '#FF6B6B'
    },
    {
      name: 'Carbohydrates',
      value: data.total_carbs * 4,
      percentage: Math.round((data.total_carbs * 4 / totalCalories) * 100),
      color: '#4ECDC4'
    },
    {
      name: 'Fats',
      value: data.total_fats * 9,
      percentage: Math.round((data.total_fats * 9 / totalCalories) * 100),
      color: '#45B7D1'
    }
  ];

  // Food items calorie breakdown
  const foodCaloriesData = data.items.map(item => ({
    name: item.name,
    calories: item.calories,
    protein: item.protein,
    carbs: item.carbs,
    fats: item.fats
  }));

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="chart-tooltip">
          <p className="tooltip-label">{label}</p>
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

  return (
    <div className="nutrition-charts">
      <h2>Nutritional Analysis Charts</h2>
      
      <div className="charts-grid">
        {/* Macronutrient Distribution Pie Chart */}
        <div className="chart-card">
          <h3>Macronutrient Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={macroData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percentage }) => `${name}: ${percentage}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {macroData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
          <div className="macro-legend">
            {macroData.map((macro, index) => (
              <div key={index} className="legend-item">
                <div 
                  className="legend-color" 
                  style={{ backgroundColor: macro.color }}
                ></div>
                <span>{macro.name}: {macro.percentage}%</span>
              </div>
            ))}
          </div>
        </div>

        {/* Food Items Calorie Breakdown */}
        <div className="chart-card">
          <h3>Calorie Breakdown by Food Item</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={foodCaloriesData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="name" 
                angle={-45}
                textAnchor="end"
                height={80}
              />
              <YAxis />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="calories" fill="#FF6B6B" name="Calories" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

export default NutritionCharts;