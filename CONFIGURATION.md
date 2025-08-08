# AI Calorie App Configuration Guide

This document explains how to configure the AI Calorie App using the centralized configuration system.

## Overview

All hardcoded values have been removed and replaced with a centralized configuration system located in `lib/config.ts`. This allows for easy customization and environment-specific settings.

## Configuration Structure

The configuration is organized into the following sections:

### API Configuration (`config.api`)
- `timeout`: API request timeout in milliseconds (default: 30000)
- `healthCheckTimeout`: Health check timeout in milliseconds (default: 5000)
- `baseUrl`: Base URL for the Python API (default: from NEXT_PUBLIC_API_URL)
- `nextjsApiBase`: Base path for Next.js API routes (default: '/api')
- `retryAttempts`: Number of retry attempts for failed requests (default: 3)

### Upload Configuration (`config.upload`)
- `maxFileSize`: Maximum file size in bytes (default: 10MB)
- `allowedTypes`: Array of allowed MIME types (default: ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'])
- `maxFileSizeMB`: Maximum file size for display purposes (default: 10)

### Analysis Configuration (`config.analysis`)
- `progressUpdateInterval`: Progress bar update interval in milliseconds (default: 200)
- `maxProgressIncrement`: Maximum progress increment per update (default: 30)
- `defaultConfidence`: Default confidence percentage (default: 96.2)
- `estimatedAnalysisTime`: Estimated analysis time in seconds (default: 12)
- `modelCount`: Number of AI models used (default: 4)

### UI Configuration (`config.ui`)
- `animationDuration`: Animation duration in milliseconds (default: 200)
- `transitionDuration`: Transition duration in milliseconds (default: 300)
- `progressBarHeight`: Progress bar height (default: '8px')
- `cardPadding`: Card padding (default: '24px')
- `borderRadius`: Border radius (default: '12px')

### Nutrition Configuration (`config.nutrition`)
- `defaultCalorieTarget`: Default daily calorie target (default: 2000)
- `macroCaloriesPerGram`: Calories per gram for macronutrients
  - `protein`: 4 calories per gram
  - `carbs`: 4 calories per gram
  - `fats`: 9 calories per gram
- `defaultPortionSizes`: Calorie thresholds for portion sizes
  - `small`: 200 calories
  - `medium`: 400 calories
  - `large`: 600 calories

### Mock Data Configuration (`config.mockData`)
- `analysisTime`: Display text for analysis time (default: '~12 seconds')
- `confidence`: Display text for confidence (default: '96.2%')
- `defaultCalories`: Default calorie value for unknown foods (default: 400)
- `sampleFoodItems`: Array of sample food items with nutritional data

## Environment-Specific Configuration

The configuration system supports different settings for different environments:

### Development
- Longer API timeout (60 seconds) for debugging
- Slower progress updates (300ms) for better visibility

### Production
- Longer timeout (45 seconds) for reliability
- More retry attempts (5)
- Faster progress updates (150ms)

### Test
- Shorter timeouts (10 seconds) for faster tests
- Single retry attempt
- Fast progress updates (50ms)
- Quick analysis time (2 seconds)

## Environment Variables

You can override configuration values using environment variables:

```bash
# Optional configuration overrides
NEXT_PUBLIC_MAX_FILE_SIZE_MB=10
NEXT_PUBLIC_API_TIMEOUT_MS=30000
NEXT_PUBLIC_DEFAULT_CALORIE_TARGET=2000
NEXT_PUBLIC_ANALYSIS_CONFIDENCE=96.2
```

## Usage Examples

### Accessing Configuration in Components

```typescript
import { config, getAnalysisConfig, getUploadConfig } from '../lib/config'

// Get specific configuration sections
const analysisConfig = getAnalysisConfig()
const uploadConfig = getUploadConfig()

// Use configuration values
const maxFileSize = uploadConfig.maxFileSize
const progressInterval = analysisConfig.progressUpdateInterval
```

### Updating Configuration at Runtime

```typescript
import { updateConfig } from '../lib/config'

// Update specific configuration values
updateConfig({
  analysis: {
    progressUpdateInterval: 100,
    defaultConfidence: 98.5
  }
})
```

### Validating Configuration

```typescript
import { validateConfig } from '../lib/config'

// Validate configuration on startup
if (!validateConfig()) {
  console.warn('Configuration validation failed')
}
```

## Benefits of This Approach

1. **No Hardcoded Values**: All configurable values are centralized
2. **Environment-Specific**: Different settings for dev/prod/test
3. **Type Safety**: Full TypeScript support with interfaces
4. **Runtime Updates**: Configuration can be updated dynamically
5. **Validation**: Built-in configuration validation
6. **Easy Maintenance**: Single source of truth for all settings

## Customization

To customize the application for your needs:

1. **Modify Default Values**: Edit `lib/config.ts` to change default values
2. **Add New Configuration**: Extend the `AppConfig` interface and add new sections
3. **Environment Variables**: Set environment variables to override defaults
4. **Runtime Updates**: Use `updateConfig()` to change values dynamically

## File Size and Upload Limits

The file upload limits are now configurable:

```typescript
// Current settings
maxFileSize: 10 * 1024 * 1024, // 10MB
allowedTypes: ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']

// To change limits, update the configuration:
updateConfig({
  upload: {
    maxFileSize: 20 * 1024 * 1024, // 20MB
    maxFileSizeMB: 20,
    allowedTypes: ['image/jpeg', 'image/png', 'image/webp', 'image/gif']
  }
})
```

## Performance Tuning

Adjust performance-related settings:

```typescript
// For slower devices
updateConfig({
  analysis: {
    progressUpdateInterval: 500, // Slower updates
    maxProgressIncrement: 20     // Smaller increments
  }
})

// For faster networks
updateConfig({
  api: {
    timeout: 15000,    // Shorter timeout
    retryAttempts: 2   // Fewer retries
  }
})
```

This configuration system makes the AI Calorie App highly customizable and maintainable while eliminating all hardcoded values.