import React, { useState } from 'react';
import {
  CheckCircle,
  AlertCircle,
  Clock,
  Brain,
  Lightbulb,
  Zap,
  Apple,
  BarChart3,
  TrendingUp,
  Target,
  RefreshCw,
  Globe,
  Activity,
  BarChart,
  
} from 'lucide-react';
import type { AnalysisResult } from '../types';
import GroqAnalysis from './GroqAnalysis';
import GroqSummary from './GroqSummary';
import DailyMealPlan from './DailyMealPlan';
import FoodItemReports from './FoodItemReports';
import AutoDietChat from './AutoDietChat';
import GenerateDietPlanButton from './GenerateDietPlanButton';
import './AnalysisResults.css';

interface AnalysisResultsProps {
  result: AnalysisResult;
  onReAnalyze?: () => void;
  isReAnalyzing?: boolean;
}

// Model Detection Breakdown Component
const ModelDetectionBreakdown: React.FC<{ 
  modelInfo: AnalysisResult['model_info'];
  dietChatResponse?: AnalysisResult['diet_chat_response'];
}> = ({ modelInfo, dietChatResponse }) => {
  if (!modelInfo) return null;

  const { model_performance, detailed_detections } = modelInfo;
  
  // Calculate total detections and success rate
  const totalDetections = Object.values(model_performance).reduce((sum, model) => 
    sum + (model.success ? model.detection_count : 0), 0
  );
  
  const successfulModels = Object.values(model_performance).filter(model => model.success).length;
  const totalModels = Object.keys(model_performance).length;
  const successRate = totalModels > 0 ? (successfulModels / totalModels) * 100 : 0;

  return (
    <div className="model-detection-report">
      <div className="report-header">
        <div className="report-title">
          <Globe size={24} />
          <h3>AI Model Detection Report</h3>
        </div>
        <div className="report-meta">
          <span className="report-date">{new Date().toLocaleDateString()}</span>
          <span className="report-time">{new Date().toLocaleTimeString()}</span>
        </div>
      </div>

      {/* Executive Summary */}
      <div className="executive-summary">
        <h4>Executive Summary</h4>
        <div className="summary-grid">
          <div className="summary-card">
            <div className="summary-icon">
              <Activity size={20} />
            </div>
            <div className="summary-content">
              <span className="summary-value">{totalDetections}</span>
              <span className="summary-label">Total Detections</span>
            </div>
          </div>
          <div className="summary-card">
            <div className="summary-icon">
              <CheckCircle size={20} />
            </div>
            <div className="summary-content">
              <span className="summary-value">{successRate.toFixed(0)}%</span>
              <span className="summary-label">Success Rate</span>
            </div>
          </div>
          <div className="summary-card">
            <div className="summary-icon">
              <Brain size={20} />
            </div>
            <div className="summary-content">
              <span className="summary-value">{successfulModels}/{totalModels}</span>
              <span className="summary-label">Active Models</span>
            </div>
          </div>
          <div className="summary-card">
            <div className="summary-icon">
              <Target size={20} />
            </div>
            <div className="summary-content">
              <span className="summary-value">{detailed_detections?.length || 0}</span>
              <span className="summary-label">Unique Foods</span>
            </div>
          </div>
        </div>
      </div>

      {/* Model Performance Table */}
      <div className="model-performance-section">
        <h4>Model Performance Analysis</h4>
        <div className="performance-table">
          <div className="table-header">
            <div className="header-cell">AI Model</div>
            <div className="header-cell">Status</div>
            <div className="header-cell">Detections</div>
            <div className="header-cell">Performance</div>
          </div>
          {Object.entries(model_performance).map(([modelName, performance]) => (
            <div key={modelName} className={`table-row ${performance.success ? 'success' : 'failed'}`}>
              <div className="table-cell model-name">
                <span className="model-label">{modelName.toUpperCase()}</span>
              </div>
              <div className="table-cell status">
                {performance.success ? (
                  <span className="status-badge success">
                    <CheckCircle size={14} />
                    Active
                  </span>
                ) : (
                  <span className="status-badge failed">
                    <AlertCircle size={14} />
                    Failed
                  </span>
                )}
              </div>
              <div className="table-cell detections">
                <span className="detection-count">{performance.detection_count}</span>
              </div>
              <div className="table-cell performance-bar">
                <div className="performance-indicator">
                  <div 
                    className="performance-fill" 
                    style={{ 
                      width: `${performance.success ? (performance.detection_count / Math.max(...Object.values(model_performance).map(p => p.detection_count))) * 100 : 0}%`,
                      backgroundColor: performance.success ? '#22c55e' : '#ef4444'
                    }}
                  ></div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Detailed Food Analysis */}
      {detailed_detections && detailed_detections.length > 0 && (
        <div className="food-analysis-section">
          <h4>Detailed Food Detection Analysis</h4>
          <div className="food-detection-table">
            <div className="table-header">
              <div className="header-cell">Food Item</div>
              <div className="header-cell">Detection Count</div>
              <div className="header-cell">Models Used</div>
              <div className="header-cell">Confidence</div>
            </div>
            {detailed_detections.slice(0, 10).map((detection, index) => (
              <div key={index} className="table-row">
                <div className="table-cell food-name">
                  <span className="food-label">{detection.food}</span>
                </div>
                <div className="table-cell detection-count">
                  <span className="count-badge">{detection.count}</span>
                </div>
                <div className="table-cell models-used">
                  <div className="model-tags">
                    {detection.methods.map((method, idx) => (
                      <span key={idx} className="model-tag">{method.toUpperCase()}</span>
                    ))}
                  </div>
                </div>
                <div className="table-cell confidence">
                  <span className="confidence-value">
                    {(detection.avg_confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
          {detailed_detections.length > 10 && (
            <div className="table-footer">
              <span>Showing top 10 of {detailed_detections.length} detected items</span>
            </div>
          )}
        </div>
      )}

      {/* Technical Notes */}
      <div className="technical-notes">
        <h4>Technical Notes</h4>
        <div className="notes-content">
          <p><strong>Detection Methodology:</strong> This analysis utilized an ensemble of {totalModels} AI models including YOLO (object detection), ViT (vision transformer), Swin (swin transformer), BLIP (image captioning), and CLIP (similarity scoring).</p>
          <p><strong>Confidence Scoring:</strong> Detection confidence is calculated based on model agreement and individual model confidence scores. Higher confidence indicates more reliable detections.</p>
          <p><strong>Model Agreement:</strong> Foods detected by multiple models are considered more reliable than single-model detections.</p>
        </div>
      </div>

      {/* AI Diet Chat Analysis */}
      {dietChatResponse && (
        <div className="diet-chat-section">
          <h4>🤖 AI Nutrition Assistant Analysis</h4>
          <div className="diet-chat-content">
            <div className="chat-summary">
              <div className="summary-header">
                <div className="ai-icon">🤖</div>
                <div className="summary-info">
                  <h5>Automatic Nutrition Analysis</h5>
                  <div className="confidence-indicator">
                    <span className="confidence-label">AI Confidence:</span>
                    <span className="confidence-value" style={{
                      color: dietChatResponse.confidence >= 0.8 ? '#22c55e' : 
                             dietChatResponse.confidence >= 0.6 ? '#f59e0b' : '#ef4444'
                    }}>
                      {(dietChatResponse.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              </div>
              
              <div className="ai-response">
                <div className="response-text">
                  {dietChatResponse.answer}
                </div>
              </div>

              {dietChatResponse.suggestions && dietChatResponse.suggestions.length > 0 && (
                <div className="suggestions-section">
                  <h6>💡 AI Recommendations</h6>
                  <div className="suggestions-list">
                    {dietChatResponse.suggestions.map((suggestion, index) => (
                      <div key={index} className="suggestion-item">
                        <span className="suggestion-icon">•</span>
                        <span className="suggestion-text">{suggestion}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {dietChatResponse.relatedTopics && dietChatResponse.relatedTopics.length > 0 && (
                <div className="topics-section">
                  <h6>🔗 Related Topics</h6>
                  <div className="topics-list">
                    {dietChatResponse.relatedTopics.map((topic, index) => (
                      <span key={index} className="topic-tag">{topic}</span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

const AnalysisResults: React.FC<AnalysisResultsProps> = ({ 
  result, 
  onReAnalyze, 
  isReAnalyzing = false 
}) => {
  const [generatedDietPlan, setGeneratedDietPlan] = useState<any>(null);
  // Handle case where no food is detected
  if (!result.success) {
    return (
      <div className="analysis-results">
        <div className="results-header">
          <div className="status-indicator error">
            <AlertCircle size={24} />
            <span>Analysis Complete</span>
          </div>
          {onReAnalyze && (
            <button 
              onClick={onReAnalyze} 
              disabled={isReAnalyzing}
              className="re-analyze-btn"
            >
              <RefreshCw size={16} className={isReAnalyzing ? 'spinning' : ''} />
              {isReAnalyzing ? 'Re-analyzing...' : 'Re-analyze'}
            </button>
          )}
        </div>

        <div className="results-content">
          <div className="error-message">
            <h3>No Food Items Detected</h3>
            <p>{result.analysis || 'The AI models were unable to detect any food items in the image with sufficient confidence.'}</p>

            <div className="suggestions">
              <h4>Suggestions for better results:</h4>
              <ul>
                <li>Ensure the image contains clear, well-lit food items</li>
                <li>Make sure the food is the main focus of the image</li>
                <li>Try taking the photo from a closer distance</li>
                <li>Ensure good lighting conditions</li>
                <li>Avoid blurry or low-quality images</li>
              </ul>
            </div>

            {result.insights && result.insights.length > 0 && (
              <div className="insights-section">
                <h4>AI Insights:</h4>
                <ul className="insights-list">
                  {result.insights.map((insight, index) => (
                    <li key={index} className="insight-item">
                      <span className="insight-bullet">•</span>
                      <span className="insight-text">{insight}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  // Handle successful detection
  const nutritionalData = result.nutritional_data || {
    total_calories: 0,
    total_protein: 0,
    total_carbs: 0,
    total_fats: 0,
    items: []
  };

  console.log('AnalysisResults - Full result:', result); // Debug log
  console.log('AnalysisResults - Nutritional data:', nutritionalData); // Debug log
  console.log('AnalysisResults - Diet chat response:', result.diet_chat_response); // Debug log

  // Ensure we have valid numbers for display
  const safeNutritionData = {
    total_calories: Number(nutritionalData.total_calories) || 0,
    total_protein: Number(nutritionalData.total_protein) || 0,
    total_carbs: Number(nutritionalData.total_carbs) || 0,
    total_fats: Number(nutritionalData.total_fats) || 0,
    items: Array.isArray(nutritionalData.items) ? nutritionalData.items : []
  };

  console.log('AnalysisResults - Safe nutrition data:', safeNutritionData); // Debug log

  // Check if we have any nutrition data at all
  const hasNutritionData = safeNutritionData.total_calories > 0 || 
                          safeNutritionData.total_protein > 0 || 
                          safeNutritionData.total_carbs > 0 || 
                          safeNutritionData.total_fats > 0;

  console.log('AnalysisResults - Has nutrition data:', hasNutritionData); // Debug log

  const detectedFoods = result.detected_foods || [];
  const insights = result.insights || [];

  return (
    <div className="analysis-results">
      <div className="results-header">
        <div className="status-indicator success">
          <CheckCircle size={24} />
          <span>Expert Analysis Complete</span>
        </div>
        {onReAnalyze && (
          <button 
            onClick={onReAnalyze} 
            disabled={isReAnalyzing}
            className="re-analyze-btn"
          >
            <RefreshCw size={16} className={isReAnalyzing ? 'spinning' : ''} />
            {isReAnalyzing ? 'Re-analyzing...' : 'Re-analyze'}
          </button>
        )}
      </div>

      <div className="results-content">
        {/* Model Detection Breakdown */}
        {result.model_info && (
          <ModelDetectionBreakdown 
          modelInfo={result.model_info} 
          dietChatResponse={result.diet_chat_response}
        />
        )}

        {/* Nutrition Summary */}
        <div className="nutrition-summary">
          <h3>Nutrition Summary</h3>
          {hasNutritionData ? (
          <div className="nutrition-grid">
            <div className="nutrition-item">
              <Zap size={20} />
              <div className="nutrition-content">
                  <span className="value">{safeNutritionData.total_calories}</span>
                  <span className="label">Calories</span>
              </div>
            </div>
            <div className="nutrition-item">
              <Apple size={20} />
              <div className="nutrition-content">
                  <span className="value">{safeNutritionData.total_protein}g</span>
                  <span className="label">Protein</span>
              </div>
            </div>
            <div className="nutrition-item">
              <BarChart3 size={20} />
              <div className="nutrition-content">
                  <span className="value">{safeNutritionData.total_carbs}g</span>
                  <span className="label">Carbs</span>
              </div>
            </div>
            <div className="nutrition-item">
              <TrendingUp size={20} />
              <div className="nutrition-content">
                  <span className="value">{safeNutritionData.total_fats}g</span>
                  <span className="label">Fats</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="no-nutrition-data">
              <p>No nutrition data available for the detected foods.</p>
              <p>This may be because the detected items are not in our nutrition database.</p>
            </div>
          )}
        </div>

        {/* GROQ AI Summary */}
        {result.groq_analysis ? (
          <GroqSummary 
            summary={result.groq_analysis.summary}
            healthScore={result.groq_analysis.healthScore}
            detectedFoods={detectedFoods}
          />
        ) : detectedFoods.length > 0 ? (
          <div className="basic-summary">
            <div className="summary-header">
              <div className="summary-title">
                <Brain size={24} className="summary-icon" />
                <h3>Food Detection Summary</h3>
              </div>
            </div>
            <div className="summary-content">
              <div className="summary-main">
                <div className="summary-text">
                  <Lightbulb size={20} className="insight-icon" />
                  <p>Successfully detected {detectedFoods.length} food item{detectedFoods.length !== 1 ? 's' : ''} in your image. 
                  {hasNutritionData ? ' Nutritional analysis completed with basic data.' : ' Enable GROQ API for detailed AI-powered nutrition analysis and personalized recommendations.'}</p>
                </div>
              </div>
              <div className="detected-foods-section">
                <h4>Detected Foods</h4>
                <div className="food-tags">
                  {detectedFoods.map((food, index) => (
                    <span key={index} className="food-tag">
                      {food}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        ) : null}

        {/* Nutrition Breakdown */}
        {hasNutritionData && (
          <div className="nutrition-breakdown">
            <h3>Macronutrient Breakdown</h3>
            <div className="macro-distribution">
              <div className="macro-item protein">
                <div className="macro-label">Protein</div>
                <div className="macro-bar">
                  <div 
                    className="macro-fill" 
                    style={{ 
                      width: `${(safeNutritionData.total_protein * 4 / safeNutritionData.total_calories) * 100}%`,
                      backgroundColor: '#22c55e'
                    }}
                  ></div>
                </div>
                <div className="macro-percentage">
                  {((safeNutritionData.total_protein * 4 / safeNutritionData.total_calories) * 100).toFixed(1)}%
                </div>
              </div>
              <div className="macro-item carbs">
                <div className="macro-label">Carbs</div>
                <div className="macro-bar">
                  <div 
                    className="macro-fill" 
                    style={{ 
                      width: `${(safeNutritionData.total_carbs * 4 / safeNutritionData.total_calories) * 100}%`,
                      backgroundColor: '#3b82f6'
                    }}
                  ></div>
                </div>
                <div className="macro-percentage">
                  {((safeNutritionData.total_carbs * 4 / safeNutritionData.total_calories) * 100).toFixed(1)}%
                </div>
              </div>
              <div className="macro-item fats">
                <div className="macro-label">Fats</div>
                <div className="macro-bar">
                  <div 
                    className="macro-fill" 
                    style={{ 
                      width: `${(safeNutritionData.total_fats * 9 / safeNutritionData.total_calories) * 100}%`,
                      backgroundColor: '#f59e0b'
                    }}
                  ></div>
                </div>
                <div className="macro-percentage">
                  {((safeNutritionData.total_fats * 9 / safeNutritionData.total_calories) * 100).toFixed(1)}%
                </div>
              </div>
            </div>
            <div className="nutrition-notes">
              <p>• Protein: {safeNutritionData.total_protein * 4} calories ({((safeNutritionData.total_protein * 4 / safeNutritionData.total_calories) * 100).toFixed(1)}%)</p>
              <p>• Carbs: {safeNutritionData.total_carbs * 4} calories ({((safeNutritionData.total_carbs * 4 / safeNutritionData.total_calories) * 100).toFixed(1)}%)</p>
              <p>• Fats: {safeNutritionData.total_fats * 9} calories ({((safeNutritionData.total_fats * 9 / safeNutritionData.total_calories) * 100).toFixed(1)}%)</p>
            </div>
          </div>
        )}

        {/* Nutrition Visualizations */}
        {hasNutritionData && (
          <div className="nutrition-visualizations">
            <h3>Nutrition Analytics & Visualizations</h3>
            
            {/* Bar Chart - Macronutrient Comparison */}
            <div className="chart-section">
              <h4>
                <BarChart size={20} />
                Macronutrient Distribution (Bar Chart)
              </h4>
              <div className="bar-chart-container">
                <div className="bar-chart">
                  <div className="bar-item">
                    <div className="bar-label">Protein</div>
                    <div className="bar-wrapper">
                      <div 
                        className="bar-fill protein-bar" 
                        style={{ 
                          height: `${(safeNutritionData.total_protein / Math.max(safeNutritionData.total_protein, safeNutritionData.total_carbs, safeNutritionData.total_fats)) * 100}%`
                        }}
                      >
                        <span className="bar-value">{safeNutritionData.total_protein}g</span>
                      </div>
                    </div>
                  </div>
                  <div className="bar-item">
                    <div className="bar-label">Carbs</div>
                    <div className="bar-wrapper">
                      <div 
                        className="bar-fill carbs-bar" 
                        style={{ 
                          height: `${(safeNutritionData.total_carbs / Math.max(safeNutritionData.total_protein, safeNutritionData.total_carbs, safeNutritionData.total_fats)) * 100}%`
                        }}
                      >
                        <span className="bar-value">{safeNutritionData.total_carbs}g</span>
                      </div>
                    </div>
                  </div>
                  <div className="bar-item">
                    <div className="bar-label">Fats</div>
                    <div className="bar-wrapper">
                      <div 
                        className="bar-fill fats-bar" 
                        style={{ 
                          height: `${(safeNutritionData.total_fats / Math.max(safeNutritionData.total_protein, safeNutritionData.total_carbs, safeNutritionData.total_fats)) * 100}%`
                        }}
                      >
                        <span className="bar-value">{safeNutritionData.total_fats}g</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Radar Chart - Nutrition Balance */}
            <div className="chart-section">
              <h4>
                <Target size={20} />
                Nutrition Balance Radar Chart
              </h4>
              <div className="radar-chart-container">
                <div className="radar-chart">
                  <svg viewBox="0 0 200 200" className="radar-svg">
                    {/* Radar grid lines */}
                    <circle cx="100" cy="100" r="80" fill="none" stroke="rgba(0,0,0,0.1)" strokeWidth="1"/>
                    <circle cx="100" cy="100" r="60" fill="none" stroke="rgba(0,0,0,0.1)" strokeWidth="1"/>
                    <circle cx="100" cy="100" r="40" fill="none" stroke="rgba(0,0,0,0.1)" strokeWidth="1"/>
                    <circle cx="100" cy="100" r="20" fill="none" stroke="rgba(0,0,0,0.1)" strokeWidth="1"/>
                    
                    {/* Radar axes */}
                    <line x1="100" y1="20" x2="100" y2="180" stroke="rgba(0,0,0,0.2)" strokeWidth="1"/>
                    <line x1="20" y1="100" x2="180" y2="100" stroke="rgba(0,0,0,0.2)" strokeWidth="1"/>
                    <line x1="35" y1="35" x2="165" y2="165" stroke="rgba(0,0,0,0.2)" strokeWidth="1"/>
                    <line x1="165" y1="35" x2="35" y2="165" stroke="rgba(0,0,0,0.2)" strokeWidth="1"/>
                    
                    {/* Nutrition data points */}
                    <circle 
                      cx="100" 
                      cy={100 - (safeNutritionData.total_protein / Math.max(safeNutritionData.total_protein, safeNutritionData.total_carbs, safeNutritionData.total_fats)) * 80} 
                      r="4" 
                      fill="#22c55e"
                    />
                    <circle 
                      cx={100 + (safeNutritionData.total_carbs / Math.max(safeNutritionData.total_protein, safeNutritionData.total_carbs, safeNutritionData.total_fats)) * 80} 
                      cy="100" 
                      r="4" 
                      fill="#3b82f6"
                    />
                    <circle 
                      cx="100" 
                      cy={100 + (safeNutritionData.total_fats / Math.max(safeNutritionData.total_protein, safeNutritionData.total_carbs, safeNutritionData.total_fats)) * 80} 
                      r="4" 
                      fill="#f59e0b"
                    />
                    
                    {/* Labels */}
                    <text x="100" y="15" textAnchor="middle" fontSize="10" fill="#22c55e" fontWeight="bold">Protein</text>
                    <text x="185" y="105" textAnchor="middle" fontSize="10" fill="#3b82f6" fontWeight="bold">Carbs</text>
                    <text x="100" y="195" textAnchor="middle" fontSize="10" fill="#f59e0b" fontWeight="bold">Fats</text>
                  </svg>
                </div>
              </div>
            </div>

            {/* Stacked Area Chart - Calorie Contribution */}
            <div className="chart-section">
              <h4>
                <TrendingUp size={20} />
                Calorie Contribution (Stacked Area)
              </h4>
              <div className="stacked-area-container">
                <div className="stacked-area-chart">
                  <div className="area-item protein-area">
                    <div className="area-label">Protein Calories</div>
                    <div className="area-fill" style={{ 
                      height: `${(safeNutritionData.total_protein * 4 / safeNutritionData.total_calories) * 100}%`,
                      backgroundColor: '#22c55e'
                    }}>
                      <span className="area-value">{safeNutritionData.total_protein * 4} cal</span>
                    </div>
                  </div>
                  <div className="area-item carbs-area">
                    <div className="area-label">Carb Calories</div>
                    <div className="area-fill" style={{ 
                      height: `${(safeNutritionData.total_carbs * 4 / safeNutritionData.total_calories) * 100}%`,
                      backgroundColor: '#3b82f6'
                    }}>
                      <span className="area-value">{safeNutritionData.total_carbs * 4} cal</span>
                    </div>
                  </div>
                  <div className="area-item fats-area">
                    <div className="area-label">Fat Calories</div>
                    <div className="area-fill" style={{ 
                      height: `${(safeNutritionData.total_fats * 9 / safeNutritionData.total_calories) * 100}%`,
                      backgroundColor: '#f59e0b'
                    }}>
                      <span className="area-value">{safeNutritionData.total_fats * 9} cal</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Nutrition Insights */}
            <div className="nutrition-insights">
              <h4>
                <Lightbulb size={20} />
                AI Nutrition Insights
              </h4>
              <div className="insights-grid">
                <div className="insight-card">
                  <div className="insight-icon">
                    <Apple size={16} />
                  </div>
                  <div className="insight-content">
                    <h5>Protein Analysis</h5>
                    <p>
                      {safeNutritionData.total_protein < 20 ? 'Low protein content. Consider adding lean meats, fish, or legumes.' :
                       safeNutritionData.total_protein > 50 ? 'High protein content. Good for muscle building and satiety.' :
                       'Moderate protein content. Well-balanced for most dietary needs.'}
                    </p>
                  </div>
                </div>
                <div className="insight-card">
                  <div className="insight-icon">
                    <BarChart3 size={16} />
                  </div>
                  <div className="insight-content">
                    <h5>Carbohydrate Balance</h5>
                    <p>
                      {safeNutritionData.total_carbs < 30 ? 'Low carb content. May need more energy sources.' :
                       safeNutritionData.total_carbs > 100 ? 'High carb content. Consider balancing with protein and fats.' :
                       'Good carbohydrate balance for sustained energy.'}
                    </p>
                  </div>
                </div>
                <div className="insight-card">
                  <div className="insight-icon">
                    <TrendingUp size={16} />
                  </div>
                  <div className="insight-content">
                    <h5>Fat Composition</h5>
                    <p>
                      {safeNutritionData.total_fats < 10 ? 'Low fat content. Consider adding healthy fats for essential nutrients.' :
                       safeNutritionData.total_fats > 40 ? 'High fat content. Monitor for heart health considerations.' :
                       'Balanced fat content for optimal nutrition.'}
                    </p>
                  </div>
                </div>
                <div className="insight-card">
                  <div className="insight-icon">
                    <Zap size={16} />
                  </div>
                  <div className="insight-content">
                    <h5>Calorie Density</h5>
                    <p>
                      {safeNutritionData.total_calories < 300 ? 'Low calorie meal. Good for weight management.' :
                       safeNutritionData.total_calories > 800 ? 'High calorie meal. Consider portion control.' :
                       'Moderate calorie content. Well-balanced meal.'}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Individual Food Items with Nutrition */}
        {safeNutritionData.items && safeNutritionData.items.length > 0 && (
          <div className="food-nutrition-details">
            <h3>Food Items with Nutrition</h3>
            <div className="food-nutrition-list">
              {safeNutritionData.items.map((item, index) => (
                <div key={index} className="food-nutrition-item">
                  <div className="food-name">{item.name}</div>
                  <div className="food-nutrition-values">
                    <span className="nutrition-value">
                      <Zap size={14} />
                      {item.calories} cal
                    </span>
                    <span className="nutrition-value">
                      <Apple size={14} />
                      {item.protein}g protein
                    </span>
                    <span className="nutrition-value">
                      <BarChart3 size={14} />
                      {item.carbs}g carbs
                    </span>
                    <span className="nutrition-value">
                      <TrendingUp size={14} />
                      {item.fats}g fats
                    </span>
                  </div>
                </div>
              ))}
          </div>
        </div>
        )}

        {/* Detected Foods */}
        {detectedFoods.length > 0 && (
          <div className="detected-foods">
            <h3>Detected Food Items</h3>
            <div className="food-tags">
              {detectedFoods.map((food, index) => (
                <span key={index} className="detected-food-tag">
                  {food}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Expert Details */}
        <div className="expert-details">
          <h3>Processing Information</h3>
          <div className="detail-item">
            <Clock size={16} />
            <span className="detail-label">Processing Time:</span>
            <span className="detail-value">{result.processing_time || 0}ms</span>
          </div>
          <div className="detail-item">
            <Brain size={16} />
            <span className="detail-label">AI Model Used:</span>
            <span className="detail-value">{result.model_used || 'expert_ensemble'}</span>
          </div>
          {result.confidence && (
            <div className="detail-item">
              <Target size={16} />
              <span className="detail-label">Overall Confidence:</span>
              <span className="detail-value">{Math.round((result.confidence || 0) * 100)}%</span>
            </div>
          )}
          {result.sessionId && (
            <div className="detail-item">
              <span className="detail-label">Session ID:</span>
              <span className="detail-value session-id">{result.sessionId}</span>
            </div>
          )}
        </div>

        {/* AI Insights */}
        {insights.length > 0 && (
          <div className="insights-section">
            <h3>AI Insights</h3>
            <ul className="insights-list">
              {insights.map((insight, index) => (
                <li key={index} className="insight-item">
                  <Lightbulb size={16} />
                  <div className="content">
                    <div className="title">AI Insight</div>
                    <div className="description">{insight}</div>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* GROQ AI Analysis */}
        {result.groq_analysis && (
          <GroqAnalysis analysis={result.groq_analysis} />
        )}

        {/* Detailed Food Item Reports */}
        {result.groq_analysis?.foodItemReports && (
          <FoodItemReports foodItemReports={result.groq_analysis.foodItemReports} />
        )}

        {/* Daily Meal Plan */}
        {result.groq_analysis?.dailyMealPlan && (
          <DailyMealPlan mealPlan={result.groq_analysis.dailyMealPlan} />
        )}

        {/* Generate Diet Plan Button */}
        {detectedFoods.length > 0 && (
          <GenerateDietPlanButton
            detectedFoods={result.detectedFoods?.map((food: any) => food.name) || detectedFoods}
            nutritionalData={result.nutritional_data}
            onDietPlanGenerated={setGeneratedDietPlan}
          />
        )}

        {/* Generated Diet Plan */}
        {generatedDietPlan && (
          <DailyMealPlan mealPlan={generatedDietPlan} />
        )}

        {/* Automatic Diet Chat Response */}
        {result.diet_chat_response && (
          <AutoDietChat dietChatResponse={result.diet_chat_response} />
        )}
        


        {/* Detailed Analysis */}
        {result.analysis && (
          <div className="detailed-analysis">
            <h3>Detailed Analysis</h3>
            <div className="analysis-text">
              {result.analysis.split('\n').map((paragraph, index) => (
                <p key={index}>{paragraph}</p>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AnalysisResults;