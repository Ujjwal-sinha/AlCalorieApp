import React, { useState } from 'react';
import { AnalysisService } from '../services/AnalysisService';
import { RefreshCw, ChefHat, Calendar } from 'lucide-react';
import './GenerateDietPlanButton.css';

interface GenerateDietPlanButtonProps {
  detectedFoods: string[];
  nutritionalData?: any;
  onDietPlanGenerated?: (dietPlan: any) => void;
  disabled?: boolean;
}

const GenerateDietPlanButton: React.FC<GenerateDietPlanButtonProps> = ({
  detectedFoods,
  nutritionalData,
  onDietPlanGenerated,
  disabled = false
}) => {
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleGenerateDietPlan = async () => {
    if (detectedFoods.length === 0) {
      setError('No food items detected to generate a diet plan');
      return;
    }

    setIsGenerating(true);
    setError(null);

    try {
      const analysisService = AnalysisService.getInstance();
      const result = await analysisService.generateDietPlan(detectedFoods, nutritionalData);

      if (result.success && result.dietPlan) {
        console.log('Diet plan generated successfully:', result.dietPlan);
        onDietPlanGenerated?.(result.dietPlan);
      } else {
        console.log('Diet plan generation failed:', result);
        setError('Failed to generate diet plan. Please try again.');
      }
    } catch (err) {
      console.error('Diet plan generation error:', err);
      setError(err instanceof Error ? err.message : 'Failed to generate diet plan');
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="generate-diet-plan-container">
      <button
        className={`generate-diet-plan-btn ${isGenerating ? 'generating' : ''} ${disabled ? 'disabled' : ''}`}
        onClick={handleGenerateDietPlan}
        disabled={disabled || isGenerating || detectedFoods.length === 0}
      >
        <div className="btn-content">
          {isGenerating ? (
            <>
              <RefreshCw size={20} className="spinning" />
              <span>Generating Diet Plan...</span>
            </>
          ) : (
            <>
              <ChefHat size={20} />
              <span>Generate Diet Plan</span>
            </>
          )}
        </div>
        <div className="btn-subtitle">
          <Calendar size={14} />
          <span>Based on {detectedFoods.length} detected foods</span>
        </div>
      </button>

      {error && (
        <div className="error-message">
          <span>⚠️ {error}</span>
        </div>
      )}

      {detectedFoods.length === 0 && !disabled && (
        <div className="info-message">
          <span>ℹ️ Upload a food image first to generate a diet plan</span>
        </div>
      )}
    </div>
  );
};

export default GenerateDietPlanButton;
