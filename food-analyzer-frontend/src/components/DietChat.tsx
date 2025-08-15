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
}

const DietChat: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sampleQuestions, setSampleQuestions] = useState<string[]>([]);
  const [userHistory, setUserHistory] = useState<string[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'connecting' | 'disconnected'>('connecting');
  const [retryCount, setRetryCount] = useState(0);
  const [showProfileModal, setShowProfileModal] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    loadSampleQuestions();
    checkConnectionStatus();
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const checkConnectionStatus = async () => {
    try {
      setConnectionStatus('connecting');
      const response = await fetch('/api/diet/health', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (response.ok) {
        const data = await response.json();
        if (data.available) {
          setConnectionStatus('connected');
          setRetryCount(0);
          
          // Add welcome message if no messages exist
          if (messages.length === 0) {
            const welcomeMessage: ChatMessage = {
              id: 'welcome',
              type: 'system',
              content: 'Welcome! I\'m your AI Nutrition Assistant. I can help you with diet advice, meal planning, and nutrition questions. Feel free to ask me anything!',
              timestamp: new Date()
            };
            setMessages([welcomeMessage]);
          }
        } else {
          setConnectionStatus('disconnected');
        }
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
      const response = await fetch('/api/diet/sample-questions');
      const data = await response.json();
      if (data.questions) {
        setSampleQuestions(data.questions);
      } else {
        // Fallback to default questions if API fails
        setSampleQuestions([
          "What should I eat to lose weight healthily?",
          "How much protein do I need daily?",
          "What are the best sources of vitamin D?",
          "How can I improve my gut health?",
          "What's a good breakfast for energy?",
          "How do I read nutrition labels?",
          "What foods help with muscle recovery?",
          "How can I reduce my sugar intake?",
          "What's the best diet for heart health?",
          "How do I plan healthy meals for the week?"
        ]);
      }
    } catch (error) {
      console.error('Failed to load sample questions:', error);
      // Fallback to default questions
      setSampleQuestions([
        "What should I eat to lose weight healthily?",
        "How much protein do I need daily?",
        "What are the best sources of vitamin D?",
        "How can I improve my gut health?",
        "What's a good breakfast for energy?",
        "How do I read nutrition labels?",
        "What foods help with muscle recovery?",
        "How can I reduce my sugar intake?",
        "What's the best diet for heart health?",
        "How do I plan healthy meals for the week?"
      ]);
    }
  };

  const sendMessage = async (question: string) => {
    if (!question.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: question,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    // Update user history
    const newHistory = [...userHistory, question].slice(-5); // Keep last 5 questions
    setUserHistory(newHistory);

    try {
      const response = await fetch('/api/diet/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question,
          userHistory: newHistory
        }),
      });

      if (response.ok) {
        const data = await response.json();

        const assistantMessage: ChatMessage = {
          id: (Date.now() + 1).toString(),
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
        setRetryCount(0);
      } else {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
    } catch (error) {
      console.error('Failed to send message:', error);
      
      // Increment retry count
      setRetryCount(prev => prev + 1);
      
      let errorMessage = 'Sorry, I\'m having trouble connecting. Please check your internet connection.';
      
      if (retryCount < 2) {
        errorMessage = `Connection attempt ${retryCount + 1} failed. Retrying...`;
      } else {
        errorMessage = 'I\'m experiencing technical difficulties. Please try again in a few moments or check your internet connection.';
        setConnectionStatus('disconnected');
      }

      const errorMessageObj: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: errorMessage,
        timestamp: new Date(),
        response: {
          answer: errorMessage,
          suggestions: [
            'Check your internet connection',
            'Try refreshing the page',
            'Wait a few moments and try again'
          ],
          relatedTopics: ['Technical Support', 'Connection Issues'],
          confidence: 0.1
        }
      };
      
      setMessages(prev => [...prev, errorMessageObj]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    sendMessage(inputValue);
  };

  const handleSampleQuestionClick = (question: string) => {
    sendMessage(question);
  };

  const handleRetryConnection = () => {
    checkConnectionStatus();
  };

  const handleQuickAction = (action: string) => {
    const actionQuestions = {
      'weight-loss': 'What should I eat to lose weight healthily?',
      'protein': 'How much protein do I need daily?',
      'vitamins': 'What are the best sources of vitamin D?',
      'gut-health': 'How can I improve my gut health?',
      'breakfast': 'What\'s a good breakfast for energy?',
      'meal-planning': 'How do I plan healthy meals for the week?'
    };
    
    const question = actionQuestions[action as keyof typeof actionQuestions];
    if (question) {
      sendMessage(question);
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return '#4CAF50';
    if (confidence >= 0.6) return '#FF9800';
    return '#F44336';
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="diet-chat">
      <div className="chat-header">
        <h2>🍎 AI Nutrition Assistant</h2>
        <p>Ask me anything about diet, nutrition, and healthy eating!</p>
        <div className="connection-status">
          <span className={`status-indicator ${connectionStatus}`}>
            {connectionStatus === 'connected' && '🟢 Connected'}
            {connectionStatus === 'connecting' && '🟡 Connecting...'}
            {connectionStatus === 'disconnected' && '🔴 Disconnected'}
          </span>
          {connectionStatus === 'disconnected' && (
            <button onClick={handleRetryConnection} className="retry-button">
              🔄 Retry
            </button>
          )}
        </div>
      </div>

      <div className="chat-container">
        <div className="messages-container">
          {messages.length === 0 && (
            <div className="welcome-message">
              <div className="welcome-icon">🤖</div>
              <h3>Welcome to your AI Nutrition Assistant!</h3>
              <p>I'm here to help you with all your diet and nutrition questions. Try asking me something or pick from the sample questions below.</p>
              {connectionStatus === 'disconnected' && (
                <div className="connection-warning">
                  <p>⚠️ Connection issues detected. Some features may be limited.</p>
                  <button onClick={handleRetryConnection} className="retry-button">
                    🔄 Retry Connection
                  </button>
                </div>
              )}
            </div>
          )}

          {messages.map((message) => (
            <div key={message.id} className={`message ${message.type}`}>
              <div className="message-content">
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
                              onClick={() => handleSampleQuestionClick(topic)}
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
                  <span>AI is thinking...</span>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        <div className="quick-actions">
          <h4>⚡ Quick Actions:</h4>
          <div className="action-buttons">
            <button onClick={() => handleQuickAction('weight-loss')}>Weight Loss</button>
            <button onClick={() => handleQuickAction('protein')}>Protein Needs</button>
            <button onClick={() => handleQuickAction('vitamins')}>Vitamins</button>
            <button onClick={() => handleQuickAction('gut-health')}>Gut Health</button>
            <button onClick={() => handleQuickAction('breakfast')}>Breakfast</button>
            <button onClick={() => handleQuickAction('meal-planning')}>Meal Planning</button>
          </div>
        </div>

        <div className="sample-questions">
          <h4>💭 Sample Questions:</h4>
          <div className="question-grid">
            {sampleQuestions.map((question, index) => (
              <button
                key={index}
                className="sample-question"
                onClick={() => handleSampleQuestionClick(question)}
                disabled={isLoading || connectionStatus === 'disconnected'}
              >
                {question}
              </button>
            ))}
          </div>
        </div>

        <form className="input-container" onSubmit={handleSubmit}>
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Ask me about diet, nutrition, or healthy eating..."
            disabled={isLoading || connectionStatus === 'disconnected'}
            className="message-input"
          />
          <button 
            type="submit" 
            disabled={isLoading || !inputValue.trim() || connectionStatus === 'disconnected'}
            className="send-button"
          >
            {isLoading ? '⏳' : '📤'}
          </button>
        </form>
      </div>
    </div>
  );
};

export default DietChat;
