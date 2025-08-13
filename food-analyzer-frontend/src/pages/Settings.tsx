import React, { useState } from 'react';
import { 
  User, 
  Bell, 
  Shield, 
  Palette,
  ArrowLeft,
  Save,
  Eye,
  EyeOff
} from 'lucide-react';
import { Link } from 'react-router-dom';
import Navigation from '../components/Navigation';
import './Settings.css';

const Settings: React.FC = () => {
  const [settings, setSettings] = useState({
    notifications: true,
    darkMode: true,
    autoSave: true,
    privacyMode: false,
    emailUpdates: false
  });

  const [showPassword, setShowPassword] = useState(false);

  const handleSettingChange = (key: string, value: boolean) => {
    setSettings(prev => ({
      ...prev,
      [key]: value
    }));
  };

  return (
    <div className="settings-page">
      <Navigation />
      
      <div className="settings-content">
        {/* Header */}
        <div className="settings-header">
          <Link to="/dashboard" className="back-button">
            <ArrowLeft size={20} />
            Back to Dashboard
          </Link>
          <div className="header-content">
            <h1>Settings</h1>
            <p>Customize your experience and preferences</p>
          </div>
        </div>

        <div className="settings-grid">
          {/* Profile Settings */}
          <div className="settings-section">
            <div className="section-header">
              <User size={24} />
              <h2>Profile</h2>
            </div>
            <div className="settings-form">
              <div className="form-group">
                <label>Display Name</label>
                <input type="text" placeholder="Enter your name" />
              </div>
              <div className="form-group">
                <label>Email</label>
                <input type="email" placeholder="Enter your email" />
              </div>
              <div className="form-group">
                <label>Password</label>
                <div className="password-input">
                  <input 
                    type={showPassword ? "text" : "password"} 
                    placeholder="Enter new password" 
                  />
                  <button 
                    className="password-toggle"
                    onClick={() => setShowPassword(!showPassword)}
                  >
                    {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Preferences */}
          <div className="settings-section">
            <div className="section-header">
              <Palette size={24} />
              <h2>Preferences</h2>
            </div>
            <div className="settings-options">
              <div className="setting-item">
                <div className="setting-info">
                  <h3>Dark Mode</h3>
                  <p>Use dark theme for better experience</p>
                </div>
                <label className="toggle">
                  <input
                    type="checkbox"
                    checked={settings.darkMode}
                    onChange={(e) => handleSettingChange('darkMode', e.target.checked)}
                  />
                  <span className="slider"></span>
                </label>
              </div>
              <div className="setting-item">
                <div className="setting-info">
                  <h3>Auto Save</h3>
                  <p>Automatically save analysis results</p>
                </div>
                <label className="toggle">
                  <input
                    type="checkbox"
                    checked={settings.autoSave}
                    onChange={(e) => handleSettingChange('autoSave', e.target.checked)}
                  />
                  <span className="slider"></span>
                </label>
              </div>
              <div className="setting-item">
                <div className="setting-info">
                  <h3>Privacy Mode</h3>
                  <p>Hide sensitive information</p>
                </div>
                <label className="toggle">
                  <input
                    type="checkbox"
                    checked={settings.privacyMode}
                    onChange={(e) => handleSettingChange('privacyMode', e.target.checked)}
                  />
                  <span className="slider"></span>
                </label>
              </div>
            </div>
          </div>

          {/* Notifications */}
          <div className="settings-section">
            <div className="section-header">
              <Bell size={24} />
              <h2>Notifications</h2>
            </div>
            <div className="settings-options">
              <div className="setting-item">
                <div className="setting-info">
                  <h3>Push Notifications</h3>
                  <p>Receive notifications for analysis completion</p>
                </div>
                <label className="toggle">
                  <input
                    type="checkbox"
                    checked={settings.notifications}
                    onChange={(e) => handleSettingChange('notifications', e.target.checked)}
                  />
                  <span className="slider"></span>
                </label>
              </div>
              <div className="setting-item">
                <div className="setting-info">
                  <h3>Email Updates</h3>
                  <p>Receive weekly nutrition reports</p>
                </div>
                <label className="toggle">
                  <input
                    type="checkbox"
                    checked={settings.emailUpdates}
                    onChange={(e) => handleSettingChange('emailUpdates', e.target.checked)}
                  />
                  <span className="slider"></span>
                </label>
              </div>
            </div>
          </div>

          {/* Privacy & Security */}
          <div className="settings-section">
            <div className="section-header">
              <Shield size={24} />
              <h2>Privacy & Security</h2>
            </div>
            <div className="settings-options">
              <div className="setting-item">
                <div className="setting-info">
                  <h3>Data Collection</h3>
                  <p>Allow anonymous data collection for improvements</p>
                </div>
                <label className="toggle">
                  <input type="checkbox" defaultChecked />
                  <span className="slider"></span>
                </label>
              </div>
              <div className="setting-item">
                <div className="setting-info">
                  <h3>Two-Factor Authentication</h3>
                  <p>Add an extra layer of security</p>
                </div>
                <button className="btn btn-secondary">Enable</button>
              </div>
            </div>
          </div>
        </div>

        {/* Save Button */}
        <div className="settings-actions">
          <button className="btn btn-primary">
            <Save size={20} />
            Save Changes
          </button>
        </div>
      </div>
    </div>
  );
};

export default Settings;
