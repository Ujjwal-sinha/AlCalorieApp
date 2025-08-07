# 🍱 AI Calorie App - Next.js Frontend

A beautiful, modern Next.js application with TypeScript that integrates with your existing Python food detection system. This app provides a comprehensive dashboard and seamless food analysis experience powered by advanced AI models.

## 🚀 Features

### ✨ **Modern UI/UX**
- **Beautiful Landing Page** with animated components
- **Interactive Dashboard** with real-time charts and statistics
- **Drag & Drop Food Analysis** with progress tracking
- **Responsive Design** that works on all devices
- **Dark/Light Mode** support with smooth transitions

### 🤖 **AI Integration**
- **Seamless Python Backend Integration** via FastAPI bridge
- **Real-time Analysis Progress** with step-by-step feedback
- **Comprehensive Results Display** with detailed nutritional breakdown
- **AI Visualizations** (Edge Detection, Grad-CAM, SHAP, LIME)
- **Multi-Model Support** (BLIP, YOLO, CNN, LLM)

### 📊 **Dashboard Features**
- **Weekly Nutrition Charts** with interactive visualizations
- **Macro Distribution** pie charts
- **Quick Stats** with progress indicators
- **Recent Analysis History** with accuracy metrics
- **AI Model Status** monitoring

## 🛠️ Tech Stack

- **Frontend**: Next.js 14, React 18, TypeScript
- **Styling**: Tailwind CSS with custom components
- **Animations**: Framer Motion
- **Charts**: Recharts
- **Icons**: Heroicons
- **File Upload**: React Dropzone
- **HTTP Client**: Axios
- **Backend Bridge**: FastAPI + Python

## 📦 Installation & Setup

### Prerequisites
- Node.js 18+ and npm/yarn
- Python 3.8+ with your existing food detection system
- All dependencies from your `calarieapp/requirements.txt`

### 1. Install Frontend Dependencies
```bash
# Install Node.js dependencies
npm install
# or
yarn install
```

### 2. Install Python API Bridge
```bash
# Install FastAPI bridge dependencies
pip install -r requirements_api.txt
```

### 3. Environment Setup
Create a `.env.local` file in the root directory:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_APP_NAME="AI Calorie App"
```

### 4. Start the Development Servers

#### Terminal 1: Start Python API Bridge
```bash
# This bridges your existing Python code with the Next.js frontend
python python_api_bridge.py
```
The API will be available at: http://localhost:8000

#### Terminal 2: Start Next.js Frontend
```bash
# Start the Next.js development server
npm run dev
# or
yarn dev
```
The frontend will be available at: http://localhost:3000

#### Terminal 3: (Optional) Keep your Streamlit app running
```bash
# Your existing Streamlit app (for reference/testing)
cd calarieapp
streamlit run app.py
```
The Streamlit app will be available at: http://localhost:8501

## 🏗️ Project Structure

```
├── app/                          # Next.js App Router
│   ├── globals.css              # Global styles with Tailwind
│   ├── layout.tsx               # Root layout with metadata
│   ├── page.tsx                 # Landing page with animations
│   ├── dashboard/               # Dashboard pages
│   │   └── page.tsx            # Main dashboard
│   ├── analyze/                 # Food analysis pages
│   │   └── page.tsx            # Image upload & analysis
│   └── api/                     # API routes
│       └── analyze-food/        # Food analysis endpoint
│           └── route.ts
├── components/                   # Reusable React components
├── lib/                         # Utility functions
│   └── api.ts                  # API client and utilities
├── types/                       # TypeScript type definitions
│   └── index.ts                # Main type definitions
├── calarieapp/                  # Your existing Python code
│   ├── app.py                  # Original Streamlit app
│   ├── agents.py               # AI agents
│   └── requirements.txt        # Python dependencies
├── python_api_bridge.py        # FastAPI bridge server
├── requirements_api.txt        # API bridge dependencies
├── package.json                # Node.js dependencies
├── tailwind.config.js          # Tailwind CSS configuration
├── tsconfig.json               # TypeScript configuration
└── next.config.js              # Next.js configuration
```

## 🔄 How It Works

### Integration Flow
1. **Next.js Frontend** → User uploads image via beautiful UI
2. **FastAPI Bridge** → Receives image and calls your existing Python functions
3. **Python Backend** → Uses your existing `describe_image_enhanced()` and `analyze_food_with_enhanced_prompt()`
4. **Response Chain** → Results flow back through FastAPI → Next.js → User

### Key Integration Points
- `python_api_bridge.py` imports your existing functions from `calarieapp/app.py`
- FastAPI endpoints wrap your Python functions with web API interface
- Next.js frontend calls these APIs for seamless integration
- All your existing AI models (BLIP, YOLO, etc.) work unchanged

## 🎨 UI Components

### Landing Page
- Hero section with animated text and call-to-action
- Feature showcase with icons and descriptions
- Statistics section with animated counters
- Responsive design with smooth transitions

### Dashboard
- Real-time nutrition charts and statistics
- Quick action buttons for food analysis
- Recent analysis history with accuracy metrics
- AI model status monitoring
- Weekly overview with interactive charts

### Food Analysis Page
- Drag & drop image upload with preview
- Real-time analysis progress with step indicators
- Comprehensive results display with nutrition breakdown
- AI visualizations placeholder (ready for integration)
- Error handling with helpful messages

## 🔧 Customization

### Styling
- Modify `tailwind.config.js` for custom colors and themes
- Update `app/globals.css` for global styles
- Component styles use Tailwind utility classes

### API Integration
- Update `lib/api.ts` to modify API calls
- Customize `python_api_bridge.py` for different Python integration
- Add new endpoints in `app/api/` directory

### Features
- Add new pages in `app/` directory
- Create reusable components in `components/`
- Define new types in `types/index.ts`

## 🚀 Deployment

### Frontend (Vercel/Netlify)
```bash
npm run build
npm start
```

### Backend (Docker/Cloud)
```bash
# Build and run the FastAPI bridge
uvicorn python_api_bridge:app --host 0.0.0.0 --port 8000
```

## 🤝 Integration with Existing Code

This Next.js app is designed to work seamlessly with your existing Python food detection system:

- ✅ **Zero Changes Required** to your existing `calarieapp/app.py`
- ✅ **All AI Models Preserved** (BLIP, YOLO, CNN, LLM)
- ✅ **Same Detection Logic** using your `describe_image_enhanced()` function
- ✅ **Same Analysis Logic** using your `analyze_food_with_enhanced_prompt()` function
- ✅ **Visualizations Ready** for your existing Grad-CAM, SHAP, LIME functions

## 📱 Screenshots

### Landing Page
- Modern hero section with gradient backgrounds
- Feature cards with hover animations
- Statistics section with animated counters
- Professional footer with social links

### Dashboard
- Interactive charts showing weekly nutrition data
- Quick stats with progress indicators
- Recent analysis history
- AI model status monitoring

### Food Analysis
- Drag & drop image upload interface
- Real-time progress tracking
- Comprehensive nutrition results
- Beautiful error handling

## 🎯 Next Steps

1. **Run the setup** following the installation instructions
2. **Test the integration** by uploading a food image
3. **Customize the UI** to match your brand/preferences
4. **Add AI visualizations** by integrating your existing visualization functions
5. **Deploy to production** using your preferred hosting platform

## 🔗 Links

- **Frontend**: http://localhost:3000
- **API Bridge**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Original Streamlit**: http://localhost:8501

## 👨‍💻 Developer

**Ujjwal Sinha**
- GitHub: [Ujjwal-sinha](https://github.com/Ujjwal-sinha)
- LinkedIn: [sinhaujjwal01](https://www.linkedin.com/in/sinhaujjwal01/)

---

Built with ❤️ using Next.js, TypeScript, and Advanced AI Models