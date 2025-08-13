import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { 
  Camera, 
  Brain, 
  BarChart3, 
  Zap, 
  ArrowRight, 
  Play,
  CheckCircle,
  TrendingUp,
  Heart,
  Users,
  Star,
  Shield,
  Clock,
  Target,
  Sparkles
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
      icon: <Brain size={32} />,
      title: "AI-Powered Detection",
      description: "Advanced computer vision with multiple AI models for precise food recognition"
    },
    {
      icon: <BarChart3 size={32} />,
      title: "Nutritional Analysis",
      description: "Comprehensive nutritional breakdown with detailed macro and micronutrients"
    },
    {
      icon: <Zap size={32} />,
      title: "Real-time Processing",
      description: "Instant analysis with expert ensemble detection for maximum accuracy"
    }
  ];

  const stats = [
    { number: "99.2%", label: "Accuracy Rate", icon: <Target size={24} /> },
    { number: "6+", label: "AI Models", icon: <Brain size={24} /> },
    { number: "1000+", label: "Food Items", icon: <BarChart3 size={24} /> },
    { number: "<2s", label: "Processing Time", icon: <Clock size={24} /> }
  ];

  const benefits = [
    "Track your daily nutrition intake",
    "Get detailed macro and micronutrient breakdown",
    "Monitor your health goals progress",
    "Access comprehensive food database",
    "Real-time AI-powered analysis",
    "Beautiful, intuitive interface"
  ];

  const testimonials = [
    {
      name: "Sarah Johnson",
      role: "Fitness Enthusiast",
      content: "This app has completely transformed how I track my nutrition. The AI detection is incredibly accurate!",
      rating: 5,
      avatar: "👩‍💼"
    },
    {
      name: "Mike Chen",
      role: "Health Coach",
      content: "I recommend this to all my clients. The detailed nutritional analysis is exactly what they need.",
      rating: 5,
      avatar: "👨‍💼"
    },
    {
      name: "Emma Davis",
      role: "Nutritionist",
      content: "The most advanced food recognition technology I've ever seen. It's like magic!",
      rating: 5,
      avatar: "👩‍⚕️"
    }
  ];

  return (
    <div className="landing-page">
      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-background">
          <div className="gradient-overlay"></div>
          <div className="floating-elements">
            <div className="floating-card card-1">
              <Heart size={20} />
              <span>99.2%</span>
            </div>
            <div className="floating-card card-2">
              <Users size={20} />
              <span>10K+</span>
            </div>
            <div className="floating-card card-3">
              <Star size={20} />
              <span>4.9★</span>
            </div>
          </div>
        </div>
        
        <div className="hero-content">
          <div className="hero-text">
            <div className="social-proof">
              <div className="user-avatars">
                <div className="avatar">👨‍💻</div>
                <div className="avatar">👩‍💼</div>
                <div className="avatar">👨‍⚕️</div>
                <div className="avatar-count">+2.1K</div>
              </div>
              <span className="proof-text">Join thousands of users tracking their nutrition</span>
            </div>

            <h1 className="hero-title">
              <span className="gradient-text">AI-Powered</span> Nutrition Tracking
            </h1>
            <p className="hero-subtitle">
              Transform your health journey with advanced computer vision and machine learning. 
              Get instant, accurate nutritional analysis of any meal.
            </p>

            <div className="hero-metrics">
              <div className="metric-card">
                <div className="metric-icon">
                  <Heart size={16} />
                </div>
                <div className="metric-content">
                  <span className="metric-value">99.2%</span>
                  <span className="metric-label">Accuracy</span>
                </div>
                <div className="metric-toggle active"></div>
              </div>
            </div>

            <div className="hero-buttons">
              <Link to="/dashboard" className="btn btn-primary btn-large">
                <Camera size={20} />
                Start Analyzing
                <ArrowRight size={20} />
              </Link>
              <button className="btn btn-secondary btn-large">
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
                    <div className="preview-image">
                      <div className="food-overlay">
                        <div className="food-metric">
                          <Heart size={16} />
                          <span>285 cal</span>
                        </div>
                      </div>
                    </div>
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

            {/* Floating Health Cards */}
            <div className="floating-health-cards">
              <div className="health-card pulse-card">
                <div className="card-header">
                  <Heart size={16} />
                  <span>Pulse Rate</span>
                </div>
                <div className="card-value">285</div>
                <div className="card-status">
                  <CheckCircle size={12} />
                  <span>Calories detected</span>
                </div>
                <div className="card-details">Accuracy: 99.2%</div>
              </div>

              <div className="health-card sleep-card">
                <div className="card-header">
                  <Clock size={16} />
                  <span>Processing</span>
                </div>
                <div className="card-value">1.2s</div>
                <div className="card-status">
                  <Sparkles size={12} />
                  <span>AI Analysis</span>
                </div>
                <div className="card-details">6 Models Active</div>
              </div>

              <div className="health-card activity-card">
                <div className="card-header">
                  <TrendingUp size={16} />
                  <span>Today's Analysis</span>
                </div>
                <div className="card-value">12</div>
                <div className="card-status">
                  <BarChart3 size={12} />
                  <span>Foods detected</span>
                </div>
                <div className="card-details">Avg: 3.2 items</div>
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
                <div className="stat-icon">
                  {stat.icon}
                </div>
                <div className="stat-number">{stat.number}</div>
                <div className="stat-label">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Testimonials Section */}
      <section className="testimonials-section">
        <div className="container">
          <div className="section-header">
            <h2>What Our Users Say</h2>
            <p>Join thousands of satisfied users</p>
          </div>
          <div className="testimonials-grid">
            {testimonials.map((testimonial, index) => (
              <div key={index} className="testimonial-card">
                <div className="testimonial-header">
                  <div className="testimonial-avatar">
                    {testimonial.avatar}
                  </div>
                  <div className="testimonial-info">
                    <h4>{testimonial.name}</h4>
                    <p>{testimonial.role}</p>
                  </div>
                  <div className="testimonial-rating">
                    {[...Array(testimonial.rating)].map((_, i) => (
                      <Star key={i} size={16} />
                    ))}
                  </div>
                </div>
                <p className="testimonial-content">{testimonial.content}</p>
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
              <p>Advanced food recognition powered by artificial intelligence. Transform your nutrition tracking with cutting-edge AI technology.</p>
              <div className="footer-social">
                <a href="#" className="social-link">
                  <Users size={20} />
                </a>
                <a href="#" className="social-link">
                  <Star size={20} />
                </a>
                <a href="#" className="social-link">
                  <Shield size={20} />
                </a>
              </div>
            </div>
            
            <div className="footer-links">
              <div className="footer-section">
                <h4>Product</h4>
                <Link to="/dashboard">Dashboard</Link>
                <Link to="/analysis">Analysis</Link>
                <Link to="/history">History</Link>
                <Link to="/settings">Settings</Link>
              </div>
              <div className="footer-section">
                <h4>Features</h4>
                <Link to="/about">AI Detection</Link>
                <Link to="/about">Nutrition Analysis</Link>
                <Link to="/about">Health Tracking</Link>
                <Link to="/about">Data Export</Link>
              </div>
              <div className="footer-section">
                <h4>Company</h4>
                <Link to="/about">About Us</Link>
                <Link to="/about">Privacy Policy</Link>
                <Link to="/about">Terms of Service</Link>
                <Link to="/about">Contact</Link>
              </div>
              <div className="footer-section">
                <h4>Support</h4>
                <Link to="/about">Help Center</Link>
                <Link to="/about">Documentation</Link>
                <Link to="/about">API Reference</Link>
                <Link to="/about">Status</Link>
              </div>
            </div>
          </div>
          
          <div className="footer-bottom">
            <div className="footer-bottom-content">
              <p>&copy; 2024 AI Calorie Analyzer. All rights reserved.</p>
              <div className="footer-bottom-links">
                <Link to="/about">Privacy</Link>
                <Link to="/about">Terms</Link>
                <Link to="/about">Cookies</Link>
              </div>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;
