import React, { useState } from 'react';
import DietChat from './DietChat';
import EnhancedDietChat from './EnhancedDietChat';

const DietChatDemo: React.FC = () => {
  const [useEnhanced, setUseEnhanced] = useState(false);

  return (
    <div style={{ padding: '1rem' }}>
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        marginBottom: '1rem',
        gap: '1rem'
      }}>
        <button
          onClick={() => setUseEnhanced(false)}
          style={{
            padding: '0.75rem 1.5rem',
            borderRadius: '25px',
            border: useEnhanced ? '2px solid #e9ecef' : '2px solid #4CAF50',
            background: useEnhanced ? 'white' : '#4CAF50',
            color: useEnhanced ? '#666' : 'white',
            fontWeight: '600',
            cursor: 'pointer',
            transition: 'all 0.2s ease'
          }}
        >
          🤖 Basic Diet Chat
        </button>
        
        <button
          onClick={() => setUseEnhanced(true)}
          style={{
            padding: '0.75rem 1.5rem',
            borderRadius: '25px',
            border: !useEnhanced ? '2px solid #e9ecef' : '2px solid #4CAF50',
            background: !useEnhanced ? 'white' : '#4CAF50',
            color: !useEnhanced ? '#666' : 'white',
            fontWeight: '600',
            cursor: 'pointer',
            transition: 'all 0.2s ease'
          }}
        >
          🚀 Enhanced Diet Chat
        </button>
      </div>

      <div style={{
        background: '#f8f9fa',
        padding: '1rem',
        borderRadius: '12px',
        marginBottom: '1rem',
        textAlign: 'center'
      }}>
        <h3 style={{ margin: '0 0 0.5rem 0', color: '#1a1a1a' }}>
          {useEnhanced ? '🚀 Enhanced Version Features:' : '🤖 Basic Version Features:'}
        </h3>
        <p style={{ margin: 0, color: '#666', fontSize: '0.9rem' }}>
          {useEnhanced 
            ? '✨ Food image analysis • 👤 User profiles • 📋 Quick actions • 🎯 Personalized advice'
            : '💬 AI chat • 💡 Smart suggestions • 🔗 Related topics • 📊 Confidence scoring'
          }
        </p>
      </div>

      {useEnhanced ? <EnhancedDietChat /> : <DietChat />}
    </div>
  );
};

export default DietChatDemo;