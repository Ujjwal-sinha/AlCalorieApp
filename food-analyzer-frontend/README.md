# AI Calorie Analyzer - Frontend

A modern React TypeScript frontend for the AI Calorie Analyzer application, featuring advanced food recognition and nutritional analysis.

## Features

- **Advanced Food Recognition**: Multiple AI models for accurate food detection
- **Real-time Analysis**: Instant nutritional breakdown of uploaded food images
- **Interactive Charts**: Beautiful visualizations of nutritional data using Recharts
- **History Tracking**: Keep track of all your food analyses
- **Responsive Design**: Works perfectly on desktop and mobile devices
- **Modern UI**: Clean, intuitive interface with smooth animations

## Technology Stack

- **React 18** with TypeScript
- **Vite** for fast development and building
- **Recharts** for data visualization
- **Lucide React** for beautiful icons
- **TensorFlow.js** for client-side AI processing
- **Canvas API** for image processing

## Project Structure

```
src/
├── components/           # React components
│   ├── ImageUpload.tsx   # File upload and image preview
│   ├── AnalysisResults.tsx # Display analysis results
│   ├── NutritionCharts.tsx # Charts and visualizations
│   └── HistoryView.tsx   # Analysis history
├── services/            # Business logic and API calls
│   ├── FoodDetectionService.ts # AI food detection
│   └── AnalysisService.ts     # Analysis orchestration
├── types/               # TypeScript type definitions
│   └── index.ts         # All application types
├── App.tsx              # Main application component
├── App.css              # Application styles
├── index.css            # Global styles
└── main.tsx             # Application entry point
```

## Key Components

### ImageUpload
- Drag & drop file upload
- Image preview with clear functionality
- Context input for additional meal information
- Loading states during analysis

### AnalysisResults
- Nutritional summary cards
- Detected food items with confidence scores
- Detailed analysis text
- Error handling and display

### NutritionCharts
- Macronutrient distribution pie chart
- Calorie breakdown by food item
- Interactive tooltips
- Responsive chart sizing

### HistoryView
- Chronological list of analyses
- Thumbnail previews
- Quick access to previous results
- Empty state handling

## Services

### FoodDetectionService
Converts Python food detection algorithms to TypeScript:
- **AdvancedFoodDetector**: Multi-model ensemble detection
- **Color Analysis**: RGB-based food identification
- **Pattern Detection**: Shape and texture analysis
- **Ensemble Fusion**: Combines multiple detection methods

### AnalysisService
Orchestrates the analysis process:
- Image preprocessing
- Food detection coordination
- Nutritional data calculation
- Result formatting and validation

## Type System

Comprehensive TypeScript types for:
- Food items and nutritional data
- Analysis results and metadata
- UI component props
- Service interfaces

## Styling

- **CSS Custom Properties**: Consistent theming
- **Flexbox & Grid**: Modern layout techniques
- **Responsive Design**: Mobile-first approach
- **Smooth Animations**: Enhanced user experience
- **Glass Morphism**: Modern visual effects

## Getting Started

1. **Install Dependencies**
   ```bash
   npm install
   ```

2. **Start Development Server**
   ```bash
   npm run dev
   ```

3. **Build for Production**
   ```bash
   npm run build
   ```

4. **Preview Production Build**
   ```bash
   npm run preview
   ```

## Usage

1. **Upload Image**: Drag and drop or click to select a food image
2. **Add Context**: Optionally provide additional meal information
3. **Analyze**: Click the analyze button to process the image
4. **View Results**: See detailed nutritional breakdown and charts
5. **Check History**: Review previous analyses in the history tab

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Performance Optimizations

- **Lazy Loading**: Components loaded on demand
- **Image Optimization**: Automatic resizing and compression
- **Memoization**: Prevent unnecessary re-renders
- **Bundle Splitting**: Optimized chunk sizes

## Future Enhancements

- **Offline Support**: PWA capabilities
- **Camera Integration**: Direct photo capture
- **Meal Planning**: Save and organize meals
- **Export Features**: PDF reports and data export
- **Social Sharing**: Share analysis results

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details