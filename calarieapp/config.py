#!/usr/bin/env python3
"""
Configuration file for AI Calorie Tracker
Centralizes all hardcoded values for easy customization
"""

import os
from typing import Dict, Any

# =============================================================================
# APP CONFIGURATION
# =============================================================================

class AppConfig:
    """Main application configuration"""
    
    # App metadata
    APP_NAME = "🍱 AI Calorie Tracker"
    APP_DESCRIPTION = "Track your nutrition with advanced AI-powered food analysis"
    APP_VERSION = "1.0.0"
    
    # Page configuration
    PAGE_TITLE = "🍱 AI Calorie Tracker"
    PAGE_ICON = "🍽️"
    LAYOUT = "wide"
    
    # Developer information
    DEVELOPER_NAME = "Ujjwal Sinha"
    DEVELOPER_GITHUB = "https://github.com/Ujjwal-sinha"
    DEVELOPER_LINKEDIN = "https://www.linkedin.com/in/sinhaujjwal01/"
    COPYRIGHT_YEAR = "2025"
    
    # Server configuration
    DEFAULT_PORT = 8501
    DEFAULT_HOST = "0.0.0.0"

# =============================================================================
# AI MODEL CONFIGURATION
# =============================================================================

class ModelConfig:
    """AI model configuration"""
    
    # LLM Configuration
    LLM_MODEL_NAME = "llama3-8b-8192"
    LLM_API_KEY_ENV = "GROQ_API_KEY"
    
    # BLIP Configuration
    BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
    
    # YOLO Configuration
    YOLO_MODEL_PATH = "yolov8n.pt"
    
    # Device configuration
    DEVICE = "cuda" if os.getenv("USE_CUDA", "true").lower() == "true" else "cpu"

# =============================================================================
# NUTRITION CONFIGURATION
# =============================================================================

class NutritionConfig:
    """Nutrition and calorie tracking configuration"""
    
    # Default calorie targets
    DEFAULT_CALORIE_TARGET = 2000
    MIN_CALORIE_TARGET = 1000
    MAX_CALORIE_TARGET = 5000
    
    # Nutrition units
    CALORIE_UNIT = "kcal"
    PROTEIN_UNIT = "g"
    CARBS_UNIT = "g"
    FATS_UNIT = "g"
    
    # Progress calculation
    PROGRESS_DECIMAL_PLACES = 1

# =============================================================================
# UI CONFIGURATION
# =============================================================================

class UIConfig:
    """User interface configuration"""
    
    # Navigation tabs
    TABS = [
        "📷 Food Analysis",
        "📊 History", 
        "📅 Daily Summary",
        "📈 Analytics"
    ]
    
    # Tab icons
    TAB_ICONS = [
        "camera",
        "clock-history", 
        "calendar",
        "graph-up"
    ]
    
    # Model status labels
    FINE_TUNED_LABEL = "🔧 Fine-tuned"
    MODEL_STATUS_AVAILABLE = "Available"
    MODEL_STATUS_UNAVAILABLE = "Not Available"
    
    # Animation settings
    ANIMATION_DURATION = "0.4s"
    HOVER_TRANSFORM = "translateY(-8px) scale(1.02)"
    
    # Color schemes
    PRIMARY_GRADIENT = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
    SUCCESS_GRADIENT = "linear-gradient(135deg, #4CAF50 0%, #45a049 100%)"
    ERROR_GRADIENT = "linear-gradient(135deg, #f44336 0%, #d32f2f 100%)"
    WARNING_GRADIENT = "linear-gradient(135deg, #ff9800 0%, #f57c00 100%)"
    INFO_GRADIENT = "linear-gradient(135deg, #2196F3 0%, #1976D2 100%)"

# =============================================================================
# VISUALIZATION CONFIGURATION
# =============================================================================

class VisualizationConfig:
    """Chart and visualization configuration"""
    
    # Chart dimensions
    FIGURE_SIZE = (10, 6)
    DPI = 100
    
    # Color palettes
    PIE_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    BAR_COLORS = ['#667eea', '#764ba2', '#f093fb', '#f5576c']
    
    # Chart styling
    FONT_SIZE = 12
    TITLE_FONT_SIZE = 16
    LEGEND_FONT_SIZE = 10
    
    # Animation settings
    CHART_ANIMATION_DURATION = 1000

# =============================================================================
# FILE CONFIGURATION
# =============================================================================

class FileConfig:
    """File handling configuration"""
    
    # Supported image formats
    SUPPORTED_IMAGE_FORMATS = ['png', 'jpg', 'jpeg']
    
    # File size limits (in bytes)
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    # Temporary file settings
    TEMP_DIR = "temp"
    CLEANUP_TEMP_FILES = True

# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

class EnvironmentConfig:
    """Environment-specific configuration"""
    
    # Environment variables
    ENV_FILE = ".env"
    
    # Logging configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Debug mode
    DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

# =============================================================================
# CACHE CONFIGURATION
# =============================================================================

class CacheConfig:
    """Caching configuration"""
    
    # Model cache settings
    MODEL_CACHE_TTL = 3600  # 1 hour
    MODEL_CACHE_MAX_SIZE = 100
    
    # Session cache settings
    SESSION_CACHE_TTL = 1800  # 30 minutes
    
    # Image cache settings
    IMAGE_CACHE_TTL = 300  # 5 minutes

# =============================================================================
# ERROR HANDLING CONFIGURATION
# =============================================================================

class ErrorConfig:
    """Error handling configuration"""
    
    # Error messages
    MISSING_API_KEY_MSG = "API key not found. Some features will be disabled."
    MODEL_LOAD_ERROR_MSG = "Failed to load model. Feature will be disabled."
    FILE_UPLOAD_ERROR_MSG = "Error uploading file. Please try again."
    
    # Retry settings
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds

# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

def validate_config() -> Dict[str, Any]:
    """Validate configuration and return any issues"""
    issues = []
    
    # Check required environment variables
    if not os.getenv(ModelConfig.LLM_API_KEY_ENV):
        issues.append(f"Missing {ModelConfig.LLM_API_KEY_ENV} environment variable")
    
    # Check model files
    if not os.path.exists(ModelConfig.YOLO_MODEL_PATH):
        issues.append(f"YOLO model file not found: {ModelConfig.YOLO_MODEL_PATH}")
    
    # Validate nutrition settings
    if NutritionConfig.MIN_CALORIE_TARGET >= NutritionConfig.MAX_CALORIE_TARGET:
        issues.append("Invalid calorie target range")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": []
    }

# =============================================================================
# CONFIGURATION EXPORTS
# =============================================================================

# Export all configuration classes
__all__ = [
    'AppConfig',
    'ModelConfig', 
    'NutritionConfig',
    'UIConfig',
    'VisualizationConfig',
    'FileConfig',
    'EnvironmentConfig',
    'CacheConfig',
    'ErrorConfig',
    'validate_config'
]
