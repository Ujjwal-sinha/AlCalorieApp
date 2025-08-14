import React, { useState, useEffect, useRef } from 'react';
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
  Sparkles,
  Smartphone,
  Award,
  Globe,
  Lock,
  Eye,
  Activity,
  PieChart,
  Calendar,
  ChevronRight,
  ChevronLeft,
  Quote,
  Download,
  Upload,
  Settings,
  Database,
  Cpu,
  Wifi,
  Smartphone as Phone,
  Monitor,
  Tablet
} from 'lucide-react';
import './LandingPage.css';

const LandingPage: React.FC = () => {
  const [currentFeature, setCurrentFeature] = useState(0);
  const [currentTestimonial, setCurrentTestimonial] = useState(0);
  const [isVisible, setIsVisible] = useState(false);
  const heroRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setIsVisible(true);
    
    const featureInterval = setInterval(() => {
      setCurrentFeature((prev) => (prev + 1) % 4);
    }, 4000);

    const testimonialInterval = setInterval(() => {
      setCurrentTestimonial((prev) => (prev + 1) % 3);
    }, 5000);

    return () => {
      clearInterval(featureInterval);
      clearInterval(testimonialInterval);
    };
  }, []);

  const features = [
    {
      icon: <Brain size={32} />,
      title: "Multi-Model AI Detection",
      description: "Advanced ensemble of 6+ AI models for unparalleled food recognition accuracy",
      color: "#4caf50",
      gradient: "linear-gradient(135deg, #4caf50, #66bb6a)"
    },
    {
      icon: <BarChart3 size={32} />,
      title: "Comprehensive Nutrition Analysis",
      description: "Detailed macro and micronutrient breakdown with personalized insights",
      color: "#2196f3",
      gradient: "linear-gradient(135deg, #2196f3, #42a5f5)"
    },
    {
      icon: <Zap size={32} />,
      title: "Real-time Processing",
      description: "Lightning-fast analysis with results in under 2 seconds",
      color: "#ff9800",
      gradient: "linear-gradient(135deg, #ff9800, #ffb74d)"
    },
    {
      icon: <Shield size={32} />,
      title: "Privacy-First Design",
      description: "Your data stays secure with end-to-end encryption and local processing",
      color: "#9c27b0",
      gradient: "linear-gradient(135deg, #9c27b0, #ba68c8)"
    }
  ];

  const stats = [
    { number: "99.2%", label: "Accuracy Rate", icon: <Target size={24} />, color: "#4caf50" },
    { number: "6+", label: "AI Models", icon: <Brain size={24} />, color: "#2196f3" },
    { number: "1000+", label: "Food Items", icon: <Database size={24} />, color: "#ff9800" },
    { number: "<2s", label: "Processing Time", icon: <Clock size={24} />, color: "#9c27b0" },
    { number: "50K+", label: "Users", icon: <Users size={24} />, color: "#f44336" },
    { number: "4.9★", label: "User Rating", icon: <Star size={24} />, color: "#ffc107" }
  ];

  const benefits = [
    {
      icon: <Activity size={24} />,
      title: "Smart Tracking",
      description: "Automatically track your daily nutrition intake with AI precision"
    },
    {
      icon: <PieChart size={24} />,
      title: "Detailed Analytics",
      description: "Get comprehensive macro and micronutrient breakdown"
    },
    {
      icon: <TrendingUp size={24} />,
      title: "Progress Monitoring",
      description: "Monitor your health goals with beautiful visualizations"
    },
    {
      icon: <Database size={24} />,
      title: "Vast Food Database",
      description: "Access comprehensive database with 1000+ food items"
    },
    {
      icon: <Cpu size={24} />,
      title: "AI-Powered Analysis",
      description: "Real-time analysis powered by advanced machine learning"
    },
    {
      icon: <Smartphone size={24} />,
      title: "Beautiful Interface",
      description: "Intuitive design that makes nutrition tracking enjoyable"
    }
  ];

  const testimonials = [
    {
      name: "Sarah Johnson",
      role: "Fitness Enthusiast",
      content: "This app has completely transformed how I track my nutrition. The AI detection is incredibly accurate and the interface is so intuitive!",
      rating: 5,
      avatar: "👩‍💼",
      image: "https://images.unsplash.com/photo-1494790108755-2616b612b786?w=150&h=150&fit=crop&crop=face"
    },
    {
      name: "Mike Chen",
      role: "Health Coach",
      content: "I recommend this to all my clients. The detailed nutritional analysis and beautiful visualizations help them stay motivated.",
      rating: 5,
      avatar: "👨‍💼",
      image: "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150&h=150&fit=crop&crop=face"
    },
    {
      name: "Emma Davis",
      role: "Nutritionist",
      content: "The most advanced food recognition technology I've ever seen. It's like magic! My clients love how easy it is to use.",
      rating: 5,
      avatar: "👩‍⚕️",
      image: "https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=150&h=150&fit=crop&crop=face"
    }
  ];

  const platforms = [
    { icon: <Phone size={32} />, name: "iOS App", status: "Available", color: "#007AFF" },
    { icon: <Phone size={32} />, name: "Android App", status: "Available", color: "#34A853" },
    { icon: <Monitor size={32} />, name: "Web App", status: "Available", color: "#4285F4" },
    { icon: <Tablet size={32} />, name: "iPad App", status: "Coming Soon", color: "#FF6B35" }
  ];

  const nextTestimonial = () => {
    setCurrentTestimonial((prev) => (prev + 1) % testimonials.length);
  };

  const prevTestimonial = () => {
    setCurrentTestimonial((prev) => (prev - 1 + testimonials.length) % testimonials.length);
  };

  return (
    <div className={`landing-page ${isVisible ? 'visible' : ''}`}>
      {/* Hero Section */}
      <section className="hero-section" ref={heroRef}>
        <div className="hero-background">
          <div className="animated-background">
            <div className="floating-shapes">
              <div className="shape shape-1"></div>
              <div className="shape shape-2"></div>
              <div className="shape shape-3"></div>
              <div className="shape shape-4"></div>
              <div className="shape shape-5"></div>
            </div>
            <div className="gradient-overlay"></div>
          </div>
        </div>
        
        <div className="hero-content">
          <div className="hero-text">
            <div className="hero-badge">
              <Sparkles size={16} />
              <span>AI-Powered Nutrition Analysis</span>
            </div>

            <h1 className="hero-title">
              Transform Your <span className="gradient-text">Nutrition Journey</span> with AI
            </h1>
            <p className="hero-subtitle">
              Experience the future of food tracking with our advanced AI technology. 
              Get instant, accurate nutritional analysis of any meal with 99.2% accuracy.
            </p>

            <div className="hero-stats">
              <div className="stat-item">
                <div className="stat-icon">
                  <Target size={20} />
                </div>
                <div className="stat-content">
                  <span className="stat-value">99.2%</span>
                  <span className="stat-label">Accuracy</span>
                </div>
              </div>
              <div className="stat-item">
                <div className="stat-icon">
                  <Clock size={20} />
                </div>
                <div className="stat-content">
                  <span className="stat-value">&lt;2s</span>
                  <span className="stat-label">Processing</span>
                </div>
              </div>
              <div className="stat-item">
                <div className="stat-icon">
                  <Users size={20} />
                </div>
                <div className="stat-content">
                  <span className="stat-value">50K+</span>
                  <span className="stat-label">Users</span>
                </div>
              </div>
            </div>

            <div className="hero-buttons">
              <Link to="/dashboard" className="btn btn-primary btn-large">
                <Camera size={20} />
                Start Free Analysis
                <ArrowRight size={20} />
              </Link>
              <button className="btn btn-secondary btn-large">
                <Play size={20} />
                Watch Demo
              </button>
            </div>

            <div className="hero-trust">
              <div className="trust-badges">
                <div className="badge">
                  <Shield size={16} />
                  <span>Privacy First</span>
                </div>
                <div className="badge">
                  <Lock size={16} />
                  <span>End-to-End Encrypted</span>
                </div>
                <div className="badge">
                  <Globe size={16} />
                  <span>Global Coverage</span>
                </div>
              </div>
            </div>
          </div>
          
          <div className="hero-visual">
            <div className="app-showcase">
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

              <div className="floating-cards">
                <div className="floating-card card-1">
                  <div className="card-icon">
                    <Brain size={20} />
                  </div>
                  <div className="card-content">
                    <div className="card-value">6+</div>
                    <div className="card-label">AI Models</div>
                  </div>
                </div>
                <div className="floating-card card-2">
                  <div className="card-icon">
                    <Target size={20} />
                  </div>
                  <div className="card-content">
                    <div className="card-value">99.2%</div>
                    <div className="card-label">Accuracy</div>
                  </div>
                </div>
                <div className="floating-card card-3">
                  <div className="card-icon">
                    <Clock size={20} />
                  </div>
                  <div className="card-content">
                    <div className="card-value">1.2s</div>
                    <div className="card-label">Processing</div>
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
            <div className="section-badge">
              <Sparkles size={16} />
              <span>Advanced Features</span>
            </div>
            <h2>Why Choose Our AI Analyzer?</h2>
            <p>Cutting-edge technology meets intuitive design for the ultimate nutrition tracking experience</p>
          </div>
          
          <div className="features-grid">
            {features.map((feature, index) => (
              <div 
                key={index} 
                className={`feature-card ${currentFeature === index ? 'active' : ''}`}
                style={{ '--feature-color': feature.color } as React.CSSProperties}
              >
                <div className="feature-icon-wrapper" style={{ background: feature.gradient }}>
                  {feature.icon}
                </div>
                <h3>{feature.title}</h3>
                <p>{feature.description}</p>
                <div className="feature-indicator"></div>
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
              <div key={index} className="stat-card" style={{ '--stat-color': stat.color } as React.CSSProperties}>
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

      {/* How It Works Section */}
      <section className="how-it-works-section">
        <div className="container">
          <div className="section-header">
            <div className="section-badge">
              <Play size={16} />
              <span>How It Works</span>
            </div>
            <h2>Simple Steps to Smart Nutrition</h2>
            <p>Get started in minutes with our intuitive three-step process</p>
          </div>
          
          <div className="steps-grid">
            <div className="step-card">
              <div className="step-number">01</div>
              <div className="step-icon">
                <Camera size={32} />
              </div>
              <h3>Take a Photo</h3>
              <p>Simply snap a photo of your meal using our advanced camera interface</p>
            </div>
            <div className="step-card">
              <div className="step-number">02</div>
              <div className="step-icon">
                <Brain size={32} />
              </div>
              <h3>AI Analysis</h3>
              <p>Our ensemble of AI models analyzes your food with 99.2% accuracy</p>
            </div>
            <div className="step-card">
              <div className="step-number">03</div>
              <div className="step-icon">
                <BarChart3 size={32} />
              </div>
              <h3>Get Results</h3>
              <p>Receive detailed nutritional breakdown and insights in under 2 seconds</p>
            </div>
          </div>
        </div>
      </section>

      {/* Benefits Section */}
      <section className="benefits-section">
        <div className="container">
          <div className="benefits-content">
            <div className="benefits-text">
              <div className="section-badge">
                <Award size={16} />
                <span>Benefits</span>
              </div>
              <h2>Everything You Need for Smart Nutrition Tracking</h2>
              <p>
                Our AI-powered platform provides comprehensive nutrition analysis 
                with industry-leading accuracy and speed, designed for modern health-conscious individuals.
              </p>
              <div className="benefits-grid">
                {benefits.map((benefit, index) => (
                  <div key={index} className="benefit-item">
                    <div className="benefit-icon">
                      {benefit.icon}
                    </div>
                    <div className="benefit-content">
                      <h4>{benefit.title}</h4>
                      <p>{benefit.description}</p>
                    </div>
                  </div>
                ))}
              </div>
              <Link to="/dashboard" className="btn btn-primary">
                Get Started Now
                <ArrowRight size={20} />
              </Link>
            </div>
            <div className="benefits-visual">
              <div className="dashboard-preview">
                <div className="preview-header">
                  <div className="preview-tabs">
                    <div className="tab active">Today</div>
                    <div className="tab">Week</div>
                    <div className="tab">Month</div>
                  </div>
                </div>
                <div className="preview-chart">
                  <div className="chart-container">
                    <PieChart size={48} />
                    <div className="chart-overlay">
                      <div className="chart-metric">
                        <span>Calories</span>
                        <span>1,245</span>
                      </div>
                    </div>
                  </div>
                </div>
                <div className="preview-stats">
                  <div className="stat-preview">
                    <div className="stat-icon">
                      <Activity size={16} />
                    </div>
                    <div className="stat-content">
                      <span>Protein</span>
                      <span>85g</span>
                    </div>
                  </div>
                  <div className="stat-preview">
                    <div className="stat-icon">
                      <BarChart3 size={16} />
                    </div>
                    <div className="stat-content">
                      <span>Carbs</span>
                      <span>120g</span>
                    </div>
                  </div>
                  <div className="stat-preview">
                    <div className="stat-icon">
                      <Heart size={16} />
                    </div>
                    <div className="stat-content">
                      <span>Fat</span>
                      <span>45g</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Testimonials Section */}
      <section className="testimonials-section">
        <div className="container">
          <div className="section-header">
            <div className="section-badge">
              <Quote size={16} />
              <span>Testimonials</span>
            </div>
            <h2>What Our Users Say</h2>
            <p>Join thousands of satisfied users who trust our AI for accurate nutrition analysis</p>
          </div>
          
          <div className="testimonials-container">
            <div className="testimonials-track">
              {testimonials.map((testimonial, index) => (
                <div 
                  key={index} 
                  className={`testimonial-card ${currentTestimonial === index ? 'active' : ''}`}
                >
                  <div className="testimonial-content">
                    <div className="testimonial-quote">
                      <Quote size={24} />
                    </div>
                    <p>{testimonial.content}</p>
                  </div>
                  <div className="testimonial-author">
                    <div className="author-avatar">
                      <img src={testimonial.image} alt={testimonial.name} />
                    </div>
                    <div className="author-info">
                      <h4>{testimonial.name}</h4>
                      <p>{testimonial.role}</p>
                      <div className="author-rating">
                        {[...Array(testimonial.rating)].map((_, i) => (
                          <Star key={i} size={16} />
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
            
            <div className="testimonials-controls">
              <button className="control-btn" onClick={prevTestimonial}>
                <ChevronLeft size={20} />
              </button>
              <div className="testimonial-dots">
                {testimonials.map((_, index) => (
                  <button 
                    key={index} 
                    className={`dot ${currentTestimonial === index ? 'active' : ''}`}
                    onClick={() => setCurrentTestimonial(index)}
                  />
                ))}
              </div>
              <button className="control-btn" onClick={nextTestimonial}>
                <ChevronRight size={20} />
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Platforms Section */}
      <section className="platforms-section">
        <div className="container">
          <div className="section-header">
            <div className="section-badge">
              <Smartphone size={16} />
              <span>Platforms</span>
            </div>
            <h2>Available on All Your Devices</h2>
            <p>Access your nutrition data seamlessly across all platforms</p>
          </div>
          
          <div className="platforms-grid">
            {platforms.map((platform, index) => (
              <div key={index} className="platform-card" style={{ '--platform-color': platform.color } as React.CSSProperties}>
                <div className="platform-icon">
                  {platform.icon}
                </div>
                <h3>{platform.name}</h3>
                <div className="platform-status">
                  <span className={`status ${platform.status === 'Available' ? 'available' : 'coming-soon'}`}>
                    {platform.status}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="cta-section">
        <div className="container">
          <div className="cta-content">
            <div className="cta-badge">
              <Sparkles size={16} />
              <span>Get Started</span>
            </div>
            <h2>Ready to Transform Your Nutrition Journey?</h2>
            <p>Join thousands of users who trust our AI for accurate food analysis and start your health transformation today</p>
            <div className="cta-buttons">
              <Link to="/dashboard" className="btn btn-primary btn-large">
                <Camera size={24} />
                Start Free Analysis
                <ArrowRight size={20} />
              </Link>
              <Link to="/about" className="btn btn-outline btn-large">
                Learn More
              </Link>
            </div>
            <div className="cta-trust">
              <div className="trust-item">
                <Shield size={16} />
                <span>No Credit Card Required</span>
              </div>
              <div className="trust-item">
                <Clock size={16} />
                <span>Setup in 2 Minutes</span>
              </div>
              <div className="trust-item">
                <Users size={16} />
                <span>50K+ Happy Users</span>
              </div>
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
              <p>&copy; 2025 AI Calorie Analyzer. All rights reserved.</p>
              <div className="footer-bottom-links">
                <Link to="/about">Privacy</Link>
                <Link to="/about">Terms</Link>
                <Link to="/about">Cookies</Link>
              </div>
            </div>
            <div className="footer-developer">
              <p>Built by <a href="https://github.com/Ujjwal-sinha" target="_blank" rel="noopener noreferrer">Ujjwal Sinha</a> • 
                <a href="https://github.com/Ujjwal-sinha" target="_blank" rel="noopener noreferrer">GitHub</a> • 
                <a href="https://www.linkedin.com/in/sinhaujjwal01/" target="_blank" rel="noopener noreferrer">LinkedIn</a>
              </p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;
