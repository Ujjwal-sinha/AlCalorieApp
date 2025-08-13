import React from 'react';
import { 
  Camera, 
  Brain, 
  Zap, 
  Shield, 
  Users,
  ArrowLeft,
  Github,
  Mail,
  Globe
} from 'lucide-react';
import { Link } from 'react-router-dom';
import Navigation from '../components/Navigation';
import './About.css';

const About: React.FC = () => {
  const features = [
    {
      icon: <Brain size={32} />,
      title: "AI-Powered Detection",
      description: "Advanced computer vision with multiple AI models including YOLO, ViT, Swin, BLIP, and CLIP for precise food recognition."
    },
    {
      icon: <Zap size={32} />,
      title: "Real-time Analysis",
      description: "Get instant nutritional analysis with expert ensemble detection for maximum accuracy and speed."
    },
    {
      icon: <Shield size={32} />,
      title: "Privacy First",
      description: "Your data is processed locally and securely. We prioritize your privacy and data protection."
    },
    {
      icon: <Users size={32} />,
      title: "User-Friendly",
      description: "Beautiful, intuitive interface designed for the best user experience across all devices."
    }
  ];

  const technologies = [
    "React & TypeScript",
    "Node.js & Express",
    "Python & PyTorch",
    "Computer Vision",
    "Machine Learning",
    "RESTful APIs"
  ];

  return (
    <div className="about-page">
      <Navigation />
      
      <div className="about-content">
        {/* Header */}
        <div className="about-header">
          <Link to="/" className="back-button">
            <ArrowLeft size={20} />
            Back to Home
          </Link>
          <div className="header-content">
            <h1>About AI Calorie Analyzer</h1>
            <p>Revolutionizing nutrition tracking with cutting-edge AI technology</p>
          </div>
        </div>

        {/* Mission Section */}
        <div className="mission-section">
          <div className="mission-content">
            <h2>Our Mission</h2>
            <p>
              We believe that understanding your nutrition shouldn't be complicated. 
              Our AI-powered platform makes it effortless to track your daily food intake 
              and make informed decisions about your health and wellness.
            </p>
            <p>
              By combining advanced computer vision, machine learning, and nutritional science, 
              we provide accurate, real-time analysis of any meal, helping you achieve your 
              health goals with confidence.
            </p>
          </div>
        </div>

        {/* Features Section */}
        <div className="features-section">
          <h2>Key Features</h2>
          <div className="features-grid">
            {features.map((feature, index) => (
              <div key={index} className="feature-card">
                <div className="feature-icon">
                  {feature.icon}
                </div>
                <h3>{feature.title}</h3>
                <p>{feature.description}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Technology Section */}
        <div className="technology-section">
          <h2>Technology Stack</h2>
          <p>Built with modern technologies for optimal performance and reliability</p>
          <div className="tech-grid">
            {technologies.map((tech, index) => (
              <div key={index} className="tech-item">
                {tech}
              </div>
            ))}
          </div>
        </div>

        {/* AI Models Section */}
        <div className="models-section">
          <h2>AI Models</h2>
          <p>Our expert ensemble combines multiple state-of-the-art models for unparalleled accuracy</p>
          <div className="models-grid">
            <div className="model-card">
              <h3>YOLO (You Only Look Once)</h3>
              <p>Real-time object detection for instant food identification</p>
            </div>
            <div className="model-card">
              <h3>Vision Transformer (ViT)</h3>
              <p>Advanced image classification using transformer architecture</p>
            </div>
            <div className="model-card">
              <h3>Swin Transformer</h3>
              <p>Hierarchical vision transformer for detailed analysis</p>
            </div>
            <div className="model-card">
              <h3>BLIP</h3>
              <p>Bootstrapping language-image pre-training for captioning</p>
            </div>
            <div className="model-card">
              <h3>CLIP</h3>
              <p>Contrastive language-image pre-training for zero-shot classification</p>
            </div>
            <div className="model-card">
              <h3>Language Models</h3>
              <p>Advanced language processing for context-aware analysis</p>
            </div>
          </div>
        </div>

        {/* Contact Section */}
        <div className="contact-section">
          <h2>Get in Touch</h2>
          <p>Have questions or suggestions? We'd love to hear from you!</p>
          <div className="contact-grid">
            <a href="mailto:contact@aicalorieanalyzer.com" className="contact-item">
              <Mail size={24} />
              <span>Email Us</span>
            </a>
            <a href="https://github.com/aicalorieanalyzer" className="contact-item">
              <Github size={24} />
              <span>GitHub</span>
            </a>
            <a href="https://aicalorieanalyzer.com" className="contact-item">
              <Globe size={24} />
              <span>Website</span>
            </a>
          </div>
        </div>

        {/* Footer */}
        <div className="about-footer">
          <div className="footer-content">
            <div className="footer-brand">
              <Camera size={32} />
              <h3>AI Calorie Analyzer</h3>
              <p>Powered by advanced artificial intelligence</p>
            </div>
            <div className="footer-info">
              <p>&copy; 2024 AI Calorie Analyzer. All rights reserved.</p>
              <p>Made with ❤️ for better nutrition tracking</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default About;
