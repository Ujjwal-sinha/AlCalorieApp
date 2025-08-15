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
  type: 'user' | 'assistant';
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
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    loadSampleQuestions();
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
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
      const response = await fetch('/api/analysis/diet-chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question,
          userHistory: newHistory
        }),
      });

      const data = await response.json();

      if (data.success) {
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
      } else {
        const errorMessage: ChatMessage = {
          id: (Date.now() + 1).toString(),
          type: 'assistant',
          content: 'Sorry, I encountered an error. Please try again.',
          timestamp: new Date()
        };
        setMessages(prev => [...prev, errorMessage]);
      }
    } catch (error) {
      console.error('Failed to send message:', error);
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: 'Sorry, I\'m having trouble connecting. Please check your internet connection.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
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
      </div>

      <div className="chat-container">
        <div className="messages-container">
          {messages.length === 0 && (
            <div className="welcome-message">
              <div className="welcome-icon">🤖</div>
              <h3>Welcome to your AI Nutrition Assistant!</h3>
              <p>I'm here to help you with all your diet and nutrition questions. Try asking me something or pick from the sample questions below.</p>
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

        <div className="sample-questions">
          <h4>💭 Sample Questions:</h4>
          <div className="question-grid">
            {sampleQuestions.map((question, index) => (
              <button
                key={index}
                className="sample-question"
                onClick={() => handleSampleQuestionClick(question)}
                disabled={isLoading}
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
            disabled={isLoading}
            className="message-input"
          />
          <button 
            type="submit" 
            disabled={isLoading || !inputValue.trim()}
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
