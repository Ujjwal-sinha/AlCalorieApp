import React, { useState, useEffect, useRef } from 'react';
import './DietChat.css';

interface DietResponse {
  answer: string;
  suggestions: string[];
  relatedTopics: string[];
  confidence: number;
}

interface ChatMessage {
  id: string;
  type: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  response?: DietResponse;
  imageUrl?: string;
  detectedFoods?: string[];
}

interface UserProfile {
  name?: string;
  age?: number;
  weight?: number;
  height?: number;
  goal?: 'weight_loss' | 'weight_gain' | 'muscle_building' | 'maintenance' | 'health';
  dietaryRestrictions?: string[];
  activityLevel?: 'sedentary' | 'light' | 'moderate' | 'active' | 'very_active';
}

const EnhancedDietChat: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sampleQuestions, setSampleQuestions] = useState<string[]>([]);
  const [userHistory, setUserHistory] = useState<string[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'connecting' | 'disconnected'>('connecting');
  const [userProfile, setUserProfile] = useState<UserProfile>({});
  const [showProfileModal, setShowProfileModal] = useState(false);
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    loadSampleQuestions();
    checkConnectionStatus();
    loadUserProfile();
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const loadUserProfile = () => {
    const saved = localStorage.getItem('dietChatProfile');
    if (saved) {
      setUserProfile(JSON.parse(saved));
    }
  };

  const saveUserProfile = (profile: UserProfile) => {
    localStorage.setItem('dietChatProfile', JSON.stringify(profile));
    setUserProfile(profile);
  };

  const checkConnectionStatus = async () => {
    try {
      setConnectionStatus('connecting');
      const response = await fetch('/api/analysis/diet-chat/health');
      
      if (response.ok) {
        const data = await response.json();
        setConnectionStatus(data.available ? 'connected' : 'disconnected');
      } else {
        setConnectionStatus('disconnected');
      }
    } catch (error) {
      console.error('Connection check failed:', error);
      setConnectionStatus('disconnected');
    }
  };

  const loadSampleQuestions = async () => {
    try {
      const response = await fetch('/api/analysis/diet-chat/sample-questions');
      const data = await response.json();
      if (data.success) {
        setSampleQuestions(data.questions);
      }
    } catch (error) {
      console.error('Failed to load sample questions:', error);
      setSampleQuestions([
        "What should I eat to lose weight healthily?",
        "How much protein do I need daily?",
        "Create a meal plan for muscle building",
        "What foods help with better sleep?",
        "How can I improve my gut health?"
      ]);
    }
  };

  const analyzeImage = async (file: File): Promise<{ foods: string[], nutritionalData: any }> => {
    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await fetch('/api/analysis/advanced', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        const foods = data.detected_foods?.map((food: any) => food.name || food) || [];
        return {
          foods,
          nutritionalData: data.nutritional_data || data.totalNutrition || {}
        };
      }
    } catch (error) {
      console.error('Image analysis failed:', error);
    }

    return { foods: [], nutritionalData: {} };
  };

  const sendMessage = async (question: string, imageFile?: File) => {
    if ((!question.trim() && !imageFile) || isLoading) return;

    let detectedFoods: string[] = [];
    let imageUrl = '';

    // Handle image upload and analysis
    if (imageFile) {
      imageUrl = URL.createObjectURL(imageFile);
      
      const systemMessage: ChatMessage = {
        id: Date.now().toString(),
        type: 'system',
        content: '🔍 Analyzing your food image...',
        timestamp: new Date(),
        imageUrl
      };
      setMessages(prev => [...prev, systemMessage]);

      const analysis = await analyzeImage(imageFile);
      detectedFoods = analysis.foods;

      if (detectedFoods.length > 0) {
        const detectionMessage: ChatMessage = {
          id: (Date.now() + 1).toString(),
          type: 'system',
          content: `🍽️ Detected foods: ${detectedFoods.join(', ')}`,
          timestamp: new Date(),
          detectedFoods
        };
        setMessages(prev => [...prev, detectionMessage]);
      }
    }

    // Create user message
    const userMessage: ChatMessage = {
      id: (Date.now() + 2).toString(),
      type: 'user',
      content: question || `Please analyze this food image and provide nutrition advice.`,
      timestamp: new Date(),
      imageUrl,
      detectedFoods
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setSelectedImage(null);
    setIsLoading(true);

    // Build context with user profile and detected foods
    let contextualQuestion = question;
    if (detectedFoods.length > 0) {
      contextualQuestion = `I have an image with these detected foods: ${detectedFoods.join(', ')}. ${question || 'Please provide nutrition analysis and advice for this meal.'}`;
    }

    // Add user profile context
    if (userProfile.goal || userProfile.dietaryRestrictions?.length) {
      contextualQuestion += ` My goal is ${userProfile.goal || 'general health'}`;
      if (userProfile.dietaryRestrictions?.length) {
        contextualQuestion += ` and I have these dietary restrictions: ${userProfile.dietaryRestrictions.join(', ')}`;
      }
    }

    const newHistory = [...userHistory, contextualQuestion].slice(-5);
    setUserHistory(newHistory);

    try {
      const response = await fetch('/api/analysis/diet-chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: contextualQuestion,
          context: detectedFoods.length > 0 ? `Detected foods: ${detectedFoods.join(', ')}` : undefined,
          userHistory: newHistory
        }),
      });

      if (response.ok) {
        const data = await response.json();

        if (data.success) {
          const assistantMessage: ChatMessage = {
            id: (Date.now() + 3).toString(),
            type: 'assistant',
            content: data.answer,
            timestamp: new Date(),
            response: {
              answer: data.answer,
              suggestions: data.suggestions,
              relatedTopics: data.relatedTopics,
              confidence: data.confidence
            }
          };

          setMessages(prev => [...prev, assistantMessage]);
          setConnectionStatus('connected');
        } else {
          throw new Error(data.error || 'Failed to get response');
        }
      } else {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
    } catch (error) {
      console.error('Failed to send message:', error);
      
      const errorMessage: ChatMessage = {
        id: (Date.now() + 3).toString(),
        type: 'assistant',
        content: 'Sorry, I\'m having trouble connecting. Please try again.',
        timestamp: new Date(),
        response: {
          answer: 'Connection error occurred',
          suggestions: ['Check your internet connection', 'Try again in a moment'],
          relatedTopics: ['Technical Support'],
          confidence: 0.1
        }
      };
      
      setMessages(prev => [...prev, errorMessage]);
      setConnectionStatus('disconnected');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    sendMessage(inputValue, selectedImage || undefined);
  };

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedImage(file);
    }
  };

  const handleQuickAction = (action: string) => {
    const quickActions: Record<string, string> = {
      'meal_plan': 'Create a personalized meal plan for my goals',
      'calorie_count': 'How many calories should I eat daily?',
      'recipe': 'Suggest a healthy recipe for dinner',
      'shopping_list': 'Create a healthy grocery shopping list',
      'nutrition_tips': 'Give me 5 nutrition tips for better health'
    };

    if (quickActions[action]) {
      sendMessage(quickActions[action]);
    }
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return '#4CAF50';
    if (confidence >= 0.6) return '#FF9800';
    return '#F44336';
  };

  return (
    <div className="diet-chat">
      <div className="chat-header">
        <h2>🍎 Enhanced AI Nutrition Assistant</h2>
        <p>Upload food images or ask nutrition questions!</p>
        <div className="connection-status">
          <span className={`status-indicator ${connectionStatus}`}>
            {connectionStatus === 'connected' && '🟢 Connected'}
            {connectionStatus === 'connecting' && '🟡 Connecting...'}
            {connectionStatus === 'disconnected' && '🔴 Disconnected'}
          </span>
          <button 
            onClick={() => setShowProfileModal(true)}
            className="profile-button"
          >
            👤 Profile
          </button>
        </div>
      </div>

      <div className="chat-container">
        <div className="messages-container">
          {messages.length === 0 && (
            <div className="welcome-message">
              <div className="welcome-icon">🤖</div>
              <h3>Welcome to your Enhanced AI Nutrition Assistant!</h3>
              <p>Upload food images for analysis or ask nutrition questions. I can help with meal planning, nutrition advice, and health goals.</p>
              
              <div className="quick-actions">
                <h4>Quick Actions:</h4>
                <div className="action-buttons">
                  <button onClick={() => handleQuickAction('meal_plan')}>📋 Meal Plan</button>
                  <button onClick={() => handleQuickAction('calorie_count')}>🔢 Daily Calories</button>
                  <button onClick={() => handleQuickAction('recipe')}>🍳 Recipe Ideas</button>
                  <button onClick={() => handleQuickAction('nutrition_tips')}>💡 Health Tips</button>
                </div>
              </div>
            </div>
          )}

          {messages.map((message) => (
            <div key={message.id} className={`message ${message.type}`}>
              <div className="message-content">
                {message.imageUrl && (
                  <div className="message-image">
                    <img src={message.imageUrl} alt="Uploaded food" />
                  </div>
                )}
                <div className="message-text">{message.content}</div>
                <div className="message-time">{formatTime(message.timestamp)}</div>
                
                {message.type === 'assistant' && message.response && (
                  <div className="message-details">
                    <div className="confidence-indicator">
                      <span>Confidence: </span>
                      <span 
                        className="confidence-score"
                        style={{ color: getConfidenceColor(message.response.confidence) }}
                      >
                        {(message.response.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                    
                    {message.response.suggestions.length > 0 && (
                      <div className="suggestions">
                        <h4>💡 Suggestions:</h4>
                        <ul>
                          {message.response.suggestions.map((suggestion, index) => (
                            <li key={index}>{suggestion}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                    
                    {message.response.relatedTopics.length > 0 && (
                      <div className="related-topics">
                        <h4>🔗 Related Topics:</h4>
                        <div className="topic-tags">
                          {message.response.relatedTopics.map((topic, index) => (
                            <span 
                              key={index} 
                              className="topic-tag"
                              onClick={() => sendMessage(`Tell me more about ${topic}`)}
                            >
                              {topic}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="message assistant">
              <div className="message-content">
                <div className="loading-indicator">
                  <div className="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                  <span>AI is analyzing...</span>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        <div className="sample-questions">
          <h4>💭 Sample Questions:</h4>
          <div className="question-grid">
            {sampleQuestions.slice(0, 6).map((question, index) => (
              <button
                key={index}
                className="sample-question"
                onClick={() => sendMessage(question)}
                disabled={isLoading}
              >
                {question}
              </button>
            ))}
          </div>
        </div>

        <form className="input-container" onSubmit={handleSubmit}>
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleImageSelect}
            accept="image/*"
            style={{ display: 'none' }}
          />
          
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="image-button"
            disabled={isLoading}
          >
            📷
          </button>
          
          {selectedImage && (
            <div className="selected-image-preview">
              <img src={URL.createObjectURL(selectedImage)} alt="Selected" />
              <button onClick={() => setSelectedImage(null)}>✕</button>
            </div>
          )}
          
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder={selectedImage ? "Ask about this food image..." : "Ask about nutrition, diet, or upload a food image..."}
            disabled={isLoading}
            className="message-input"
          />
          
          <button 
            type="submit" 
            disabled={isLoading || (!inputValue.trim() && !selectedImage)}
            className="send-button"
          >
            {isLoading ? '⏳' : '📤'}
          </button>
        </form>
      </div>

      {/* Profile Modal */}
      {showProfileModal && (
        <div className="modal-overlay" onClick={() => setShowProfileModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h3>👤 Your Profile</h3>
            <div className="profile-form">
              <input
                type="text"
                placeholder="Name (optional)"
                value={userProfile.name || ''}
                onChange={(e) => setUserProfile({...userProfile, name: e.target.value})}
              />
              
              <select
                value={userProfile.goal || ''}
                onChange={(e) => setUserProfile({...userProfile, goal: e.target.value as any})}
              >
                <option value="">Select your goal</option>
                <option value="weight_loss">Weight Loss</option>
                <option value="weight_gain">Weight Gain</option>
                <option value="muscle_building">Muscle Building</option>
                <option value="maintenance">Maintenance</option>
                <option value="health">General Health</option>
              </select>
              
              <input
                type="text"
                placeholder="Dietary restrictions (comma separated)"
                value={userProfile.dietaryRestrictions?.join(', ') || ''}
                onChange={(e) => setUserProfile({
                  ...userProfile, 
                  dietaryRestrictions: e.target.value.split(',').map(s => s.trim()).filter(Boolean)
                })}
              />
              
              <div className="modal-buttons">
                <button onClick={() => {
                  saveUserProfile(userProfile);
                  setShowProfileModal(false);
                }}>
                  Save Profile
                </button>
                <button onClick={() => setShowProfileModal(false)}>
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default EnhancedDietChat;