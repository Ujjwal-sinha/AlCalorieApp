import React from 'react';
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
  Activity
} from 'lucide-react';
import type { AnalysisResult } from '../types';
import './AnalysisResults.css';

interface AnalysisResultsProps {
  result: AnalysisResult;
  onReAnalyze?: () => void;
  isReAnalyzing?: boolean;
}

// Model Detection Breakdown Component
const ModelDetectionBreakdown: React.FC<{ modelInfo: AnalysisResult['model_info'] }> = ({ modelInfo }) => {
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
    </div>
  );
};

const AnalysisResults: React.FC<AnalysisResultsProps> = ({ 
  result, 
  onReAnalyze, 
  isReAnalyzing = false 
}) => {
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
          <ModelDetectionBreakdown modelInfo={result.model_info} />
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