import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { 
  Camera, 
  Brain, 
  BarChart3, 
  Zap, 
  Shield, 
  Users, 
  ArrowRight, 
  Play,
  Star,
  CheckCircle,
  TrendingUp,
  Smartphone
} from 'lucide-react';
import './LandingPage.css';

const LandingPage: React.FC = () => {
  const [currentFeature, setCurrentFeature] = useState(0);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    setIsVisible(true);
    
    const interval = setInterval(() => {
      setCurrentFeature((prev) => (prev + 1) % 3);
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  const features = [
    {
      icon: <Brain className="feature-icon" />,
      title: "AI-Powered Detection",
      description: "Advanced computer vision with multiple AI models for precise food recognition"
    },
    {
      icon: <BarChart3 className="feature-icon" />,
      title: "Nutritional Analysis",
      description: "Comprehensive nutritional breakdown with detailed macro and micronutrients"
    },
    {
      icon: <Zap className="feature-icon" />,
      title: "Real-time Processing",
      description: "Instant analysis with expert ensemble detection for maximum accuracy"
    }
  ];

  const stats = [
    { number: "99.2%", label: "Accuracy Rate" },
    { number: "6+", label: "AI Models" },
    { number: "1000+", label: "Food Items" },
    { number: "<2s", label: "Processing Time" }
  ];

  const benefits = [
    "Track your daily nutrition intake",
    "Get detailed macro and micronutrient breakdown",
    "Monitor your health goals progress",
    "Access comprehensive food database",
    "Real-time AI-powered analysis",
    "Beautiful, intuitive interface"
  ];

  return (
    <div className="landing-page">
      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-background">
          <div className="gradient-overlay"></div>
          <div className="floating-shapes">
            <div className="shape shape-1"></div>
            <div className="shape shape-2"></div>
            <div className="shape shape-3"></div>
          </div>
        </div>
        
        <div className="hero-content">
          <div className="hero-text">
            <h1 className="hero-title">
              <span className="gradient-text">AI-Powered</span> Food Analysis
            </h1>
            <p className="hero-subtitle">
              Transform your nutrition tracking with advanced computer vision and machine learning. 
              Get instant, accurate nutritional analysis of any meal.
            </p>
            <div className="hero-buttons">
              <Link to="/dashboard" className="btn btn-primary">
                <Camera size={20} />
                Start Analyzing
                <ArrowRight size={20} />
              </Link>
              <button className="btn btn-secondary">
                <Play size={20} />
                Watch Demo
              </button>
            </div>
          </div>
          
          <div className="hero-visual">
            <div className="phone-mockup">
              <div className="phone-screen">
                <div className="app-preview">
                  <div className="preview-header">
                    <div className="preview-logo">
                      <Camera size={24} />
                    </div>
                    <span>AI Calorie Analyzer</span>
                  </div>
                  <div className="preview-content">
                    <div className="preview-image"></div>
                    <div className="preview-results">
                      <div className="result-item">
                        <span>🍕 Pizza</span>
                        <span>285 cal</span>
                      </div>
                      <div className="result-item">
                        <span>🥗 Salad</span>
                        <span>120 cal</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="features-section">
        <div className="container">
          <div className="section-header">
            <h2>Why Choose Our AI Analyzer?</h2>
            <p>Cutting-edge technology meets intuitive design</p>
          </div>
          
          <div className="features-grid">
            {features.map((feature, index) => (
              <div 
                key={index} 
                className={`feature-card ${currentFeature === index ? 'active' : ''}`}
              >
                <div className="feature-icon-wrapper">
                  {feature.icon}
                </div>
                <h3>{feature.title}</h3>
                <p>{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="stats-section">
        <div className="container">
          <div className="stats-grid">
            {stats.map((stat, index) => (
              <div key={index} className="stat-card">
                <div className="stat-number">{stat.number}</div>
                <div className="stat-label">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Benefits Section */}
      <section className="benefits-section">
        <div className="container">
          <div className="benefits-content">
            <div className="benefits-text">
              <h2>Everything You Need for Smart Nutrition Tracking</h2>
              <p>
                Our AI-powered platform provides comprehensive nutrition analysis 
                with industry-leading accuracy and speed.
              </p>
              <ul className="benefits-list">
                {benefits.map((benefit, index) => (
                  <li key={index}>
                    <CheckCircle size={20} />
                    {benefit}
                  </li>
                ))}
              </ul>
              <Link to="/dashboard" className="btn btn-primary">
                Get Started Now
                <ArrowRight size={20} />
              </Link>
            </div>
            <div className="benefits-visual">
              <div className="dashboard-preview">
                <div className="preview-chart">
                  <TrendingUp size={48} />
                </div>
                <div className="preview-stats">
                  <div className="stat-preview">
                    <span>Calories</span>
                    <span>1,245</span>
                  </div>
                  <div className="stat-preview">
                    <span>Protein</span>
                    <span>85g</span>
                  </div>
                  <div className="stat-preview">
                    <span>Carbs</span>
                    <span>120g</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="cta-section">
        <div className="container">
          <div className="cta-content">
            <h2>Ready to Transform Your Nutrition Journey?</h2>
            <p>Join thousands of users who trust our AI for accurate food analysis</p>
            <div className="cta-buttons">
              <Link to="/dashboard" className="btn btn-primary btn-large">
                <Camera size={24} />
                Start Free Analysis
              </Link>
              <Link to="/about" className="btn btn-outline btn-large">
                Learn More
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="landing-footer">
        <div className="container">
          <div className="footer-content">
            <div className="footer-brand">
              <div className="footer-logo">
                <Camera size={32} />
                <span>AI Calorie Analyzer</span>
              </div>
              <p>Advanced food recognition powered by artificial intelligence</p>
            </div>
            <div className="footer-links">
              <div className="footer-section">
                <h4>Product</h4>
                <Link to="/dashboard">Dashboard</Link>
                <Link to="/analysis">Analysis</Link>
                <Link to="/history">History</Link>
              </div>
              <div className="footer-section">
                <h4>Company</h4>
                <Link to="/about">About</Link>
                <Link to="/settings">Settings</Link>
              </div>
            </div>
          </div>
          <div className="footer-bottom">
            <p>&copy; 2024 AI Calorie Analyzer. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;
