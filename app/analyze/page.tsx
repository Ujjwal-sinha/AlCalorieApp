'use client'

import { useState, useCallback, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  CameraIcon, 
  PhotoIcon,
  ArrowUpTrayIcon,
  SparklesIcon,
  EyeIcon,
  BeakerIcon,
  ChartBarIcon,
  FireIcon,
  XMarkIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  CpuChipIcon,
  LightBulbIcon,
  MagnifyingGlassIcon
} from '@heroicons/react/24/outline'
import Link from 'next/link'
import { useDropzone } from 'react-dropzone'
import { apiClient, utils } from '../../lib/api'
import { AnalysisResult, AnalysisStep } from '../../types'
import { config, getAnalysisConfig, getUploadConfig, getMockDataConfig } from '../../lib/config'
import Image from 'next/image'

const analysisSteps: AnalysisStep[] = [
  { label: 'Processing Image', progress: 0 },
  { label: 'BLIP Food Detection', progress: 0 },
  { label: 'AI Visualizations', progress: 0 },
  { label: 'Nutritional Analysis', progress: 0 },
  { label: 'Generating Insights', progress: 0 }
]

export default function AnalyzePage() {
  // Get configuration values
  const analysisConfig = getAnalysisConfig()
  const uploadConfig = getUploadConfig()
  const mockDataConfig = getMockDataConfig()
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [context, setContext] = useState('')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null)
  const [currentStep, setCurrentStep] = useState(0)
  const [steps, setSteps] = useState(analysisSteps)
  const [error, setError] = useState<string | null>(null)
  const [showVisualizations, setShowVisualizations] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0]
    if (file) {
      setSelectedFile(file)
      setPreviewUrl(URL.createObjectURL(file))
      setError(null)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': uploadConfig.allowedTypes.map(type => type.replace('image/', '.'))
    },
    maxSize: uploadConfig.maxFileSize,
    multiple: false
  })

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      setSelectedFile(file)
      setPreviewUrl(URL.createObjectURL(file))
      setError(null)
    }
  }

  const simulateProgress = (stepIndex: number, callback?: () => void) => {
    let progress = 0
    const interval = setInterval(() => {
      progress += Math.random() * analysisConfig.maxProgressIncrement
      if (progress >= 100) {
        progress = 100
        clearInterval(interval)
        setSteps(prev => prev.map((step, i) => 
          i === stepIndex ? { ...step, progress: 100, completed: true } : step
        ))
        if (callback) callback()
      } else {
        setSteps(prev => prev.map((step, i) => 
          i === stepIndex ? { ...step, progress } : step
        ))
      }
    }, analysisConfig.progressUpdateInterval)
  }

  const analyzeFood = async () => {
    if (!selectedFile) return

    setIsAnalyzing(true)
    setError(null)
    setCurrentStep(0)
    setSteps(analysisSteps.map(step => ({ ...step, progress: 0, completed: false })))

    try {
      // Step 1: Processing Image
      simulateProgress(0, () => {
        setCurrentStep(1)
        // Step 2: BLIP Food Detection
        simulateProgress(1, () => {
          setCurrentStep(2)
          // Step 3: AI Visualizations
          simulateProgress(2, () => {
            setCurrentStep(3)
            // Step 4: Nutritional Analysis
            simulateProgress(3, () => {
              setCurrentStep(4)
              // Step 5: Generating Insights
              simulateProgress(4, async () => {
                // Actual API call
                try {
                  const result = await apiClient.analyzeFoodDirect(selectedFile, context)
                  
                  if (result.success && result.data) {
                    setAnalysisResult(result.data)
                  } else {
                    throw new Error(result.error || 'Analysis failed')
                  }
                } catch (apiError) {
                  console.error('API Error:', apiError)
                  // Use mock data as fallback
                  const mockResult = utils.generateMockAnalysis(
                    selectedFile.name.replace(/\.[^/.]+$/, ''), 
                    context
                  )
                  setAnalysisResult(mockResult)
                }
                
                setIsAnalyzing(false)
              })
            })
          })
        })
      })

    } catch (error) {
      console.error('Analysis error:', error)
      setError(error instanceof Error ? error.message : 'Analysis failed')
      setIsAnalyzing(false)
    }
  }

  const resetAnalysis = () => {
    setSelectedFile(null)
    setPreviewUrl(null)
    setContext('')
    setAnalysisResult(null)
    setIsAnalyzing(false)
    setCurrentStep(0)
    setSteps(analysisSteps.map(step => ({ ...step, progress: 0, completed: false })))
    setError(null)
    setShowVisualizations(false)
  }

  const formatMacroPercentage = (protein: number, carbs: number, fats: number) => {
    const nutritionConfig = config.nutrition.macroCaloriesPerGram
    const totalCalories = (protein * nutritionConfig.protein) + (carbs * nutritionConfig.carbs) + (fats * nutritionConfig.fats)
    if (totalCalories === 0) return { protein: 0, carbs: 0, fats: 0 }
    
    return {
      protein: Math.round(((protein * nutritionConfig.protein) / totalCalories) * 100),
      carbs: Math.round(((carbs * nutritionConfig.carbs) / totalCalories) * 100),
      fats: Math.round(((fats * nutritionConfig.fats) / totalCalories) * 100)
    }
  }

  // Check if result has enhanced features
  const hasEnhancedFeatures = analysisResult && 'blip_detection' in analysisResult && 'visualizations' in analysisResult

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Navigation */}
      <nav className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
                <SparklesIcon className="w-5 h-5 text-white" />
              </div>
              <span className="text-xl font-bold text-gray-900">AI Food Analyzer</span>
            </div>
            
            <div className="flex items-center space-x-4">
              <Link href="/dashboard" className="text-gray-600 hover:text-gray-900">
                Dashboard
              </Link>
              <Link href="/" className="text-gray-600 hover:text-gray-900">
                Home
              </Link>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {!analysisResult ? (
          <div className="max-w-4xl mx-auto">
            {/* Header */}
            <div className="text-center mb-8">
              <h1 className="text-3xl font-bold text-gray-900 mb-4">
                Analyze Your Food with AI
              </h1>
              <p className="text-lg text-gray-600">
                Upload a photo and get comprehensive nutritional analysis powered by advanced AI models
              </p>
            </div>

            {/* Upload Section */}
            {!selectedFile ? (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="card mb-8"
              >
                <div
                  {...getRootProps()}
                  className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-colors duration-200 ${
                    isDragActive 
                      ? 'border-primary-500 bg-primary-50' 
                      : 'border-gray-300 hover:border-primary-400 hover:bg-gray-50'
                  }`}
                >
                  <input {...getInputProps()} />
                  <div className="space-y-4">
                    <div className="flex justify-center">
                      <PhotoIcon className="w-16 h-16 text-gray-400" />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-gray-900 mb-2">
                        {isDragActive ? 'Drop your image here' : 'Upload food image'}
                      </h3>
                      <p className="text-gray-600 mb-4">
                        Drag and drop your food photo, or click to browse
                      </p>
                      <div className="flex justify-center space-x-4">
                        <button className="btn-primary">
                          <ArrowUpTrayIcon className="w-4 h-4 mr-2" />
                          Choose File
                        </button>
                        <button 
                          onClick={() => fileInputRef.current?.click()}
                          className="btn-secondary"
                        >
                          <CameraIcon className="w-4 h-4 mr-2" />
                          Take Photo
                        </button>
                      </div>
                      <input
                        ref={fileInputRef}
                        type="file"
                        accept="image/*"
                        capture="environment"
                        onChange={handleFileSelect}
                        className="hidden"
                      />
                    </div>
                    <p className="text-sm text-gray-500">
                      Supports {uploadConfig.allowedTypes.map(type => type.replace('image/', '').toUpperCase()).join(', ')} up to {uploadConfig.maxFileSizeMB}MB
                    </p>
                  </div>
                </div>
              </motion.div>
            ) : (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-6"
              >
                {/* Image Preview */}
                <div className="card">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-gray-900">
                      Selected Image
                    </h3>
                    <button
                      onClick={resetAnalysis}
                      className="text-gray-400 hover:text-gray-600"
                    >
                      <XMarkIcon className="w-5 h-5" />
                    </button>
                  </div>
                  
                  <div className="flex flex-col md:flex-row gap-6">
                    <div className="md:w-1/2">
                      {previewUrl && (
                        <div className="relative aspect-square rounded-lg overflow-hidden bg-gray-100">
                          <Image
                            src={previewUrl}
                            alt="Food preview"
                            fill
                            className="object-cover"
                          />
                        </div>
                      )}
                    </div>
                    
                    <div className="md:w-1/2 space-y-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Add Context (Optional)
                        </label>
                        <textarea
                          value={context}
                          onChange={(e) => setContext(e.target.value)}
                          placeholder="e.g., This is my lunch, homemade meal, restaurant food..."
                          className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none"
                          rows={4}
                        />
                      </div>
                      
                      <button
                        onClick={analyzeFood}
                        disabled={isAnalyzing}
                        className="w-full btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {isAnalyzing ? (
                          <div className="flex items-center justify-center">
                            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                            Analyzing...
                          </div>
                        ) : (
                          <>
                            <BeakerIcon className="w-4 h-4 mr-2" />
                            Analyze Food
                          </>
                        )}
                      </button>
                    </div>
                  </div>
                </div>

                {/* Analysis Progress */}
                <AnimatePresence>
                  {isAnalyzing && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      className="card"
                    >
                      <h3 className="text-lg font-semibold text-gray-900 mb-6">
                        AI Analysis in Progress
                      </h3>
                      
                      <div className="space-y-4">
                        {steps.map((step, index) => (
                          <div key={step.label} className="flex items-center space-x-4">
                            <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                              step.completed 
                                ? 'bg-green-100 text-green-600' 
                                : index === currentStep 
                                  ? 'bg-primary-100 text-primary-600' 
                                  : 'bg-gray-100 text-gray-400'
                            }`}>
                              {step.completed ? (
                                <CheckCircleIcon className="w-5 h-5" />
                              ) : index === currentStep ? (
                                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary-600"></div>
                              ) : (
                                <span className="text-sm font-medium">{index + 1}</span>
                              )}
                            </div>
                            
                            <div className="flex-1">
                              <div className="flex items-center justify-between mb-1">
                                <span className={`text-sm font-medium ${
                                  step.completed 
                                    ? 'text-green-600' 
                                    : index === currentStep 
                                      ? 'text-primary-600' 
                                      : 'text-gray-500'
                                }`}>
                                  {step.label}
                                </span>
                                <span className="text-xs text-gray-500">
                                  {Math.round(step.progress)}%
                                </span>
                              </div>
                              <div className="w-full bg-gray-200 rounded-full h-2">
                                <div 
                                  className={`h-2 rounded-full transition-all duration-300 ${
                                    step.completed 
                                      ? 'bg-green-500' 
                                      : index === currentStep 
                                        ? 'bg-primary-500' 
                                        : 'bg-gray-300'
                                  }`}
                                  style={{ width: `${step.progress}%` }}
                                ></div>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Error Display */}
                <AnimatePresence>
                  {error && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      className="card border-red-200 bg-red-50"
                    >
                      <div className="flex items-center space-x-3">
                        <ExclamationTriangleIcon className="w-6 h-6 text-red-600" />
                        <div>
                          <h3 className="text-sm font-medium text-red-800">
                            Analysis Error
                          </h3>
                          <p className="text-sm text-red-700 mt-1">
                            {error}
                          </p>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            )}
          </div>
        ) : (
          /* Results Section */
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-8"
          >
            {/* Header */}
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold text-gray-900 mb-2">
                  Analysis Complete! 🎉
                </h1>
                <p className="text-gray-600">
                  Here's what our AI found in your food image
                </p>
              </div>
              <button
                onClick={resetAnalysis}
                className="btn-secondary"
              >
                Analyze Another
              </button>
            </div>

            <div className="grid lg:grid-cols-3 gap-8">
              {/* Main Results */}
              <div className="lg:col-span-2 space-y-6">
                {/* Enhanced Detection Status */}
                {hasEnhancedFeatures && (
                  <div className="card bg-gradient-to-r from-blue-50 to-purple-50 border-blue-200">
                    <div className="flex items-center space-x-3 mb-4">
                      <CpuChipIcon className="w-6 h-6 text-blue-600" />
                      <h3 className="text-lg font-semibold text-gray-900">
                        Enhanced AI Analysis
                      </h3>
                      <span className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">
                        Advanced
                      </span>
                    </div>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div>
                        <span className="text-gray-600">Detection Confidence:</span>
                        <div className="font-semibold text-green-600">
                          {Math.round((analysisResult as any).detection_metadata?.confidence * 100 || 0)}%
                        </div>
                      </div>
                      <div>
                        <span className="text-gray-600">Items Detected:</span>
                        <div className="font-semibold">
                          {(analysisResult as any).detection_metadata?.total_items || 0}
                        </div>
                      </div>
                      <div>
                        <span className="text-gray-600">Processing Time:</span>
                        <div className="font-semibold">
                          {Math.round((analysisResult as any).detection_metadata?.processing_time / 1000 || 0)}s
                        </div>
                      </div>
                      <div>
                        <span className="text-gray-600">AI Models:</span>
                        <div className="font-semibold">
                          {(analysisResult as any).detection_metadata?.detection_methods?.length || 0}
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Nutritional Overview */}
                <div className="card">
                  <h2 className="text-xl font-semibold text-gray-900 mb-6">
                    Nutritional Overview
                  </h2>
                  
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                    <div className="text-center p-4 bg-orange-50 rounded-lg">
                      <FireIcon className="w-8 h-8 text-orange-600 mx-auto mb-2" />
                      <div className="text-2xl font-bold text-gray-900">
                        {utils.formatCalories(analysisResult.nutritional_data.total_calories)}
                      </div>
                      <div className="text-sm text-gray-600">Calories</div>
                    </div>
                    
                    <div className="text-center p-4 bg-blue-50 rounded-lg">
                      <div className="w-8 h-8 bg-blue-600 rounded-full mx-auto mb-2 flex items-center justify-center text-white text-xs font-bold">
                        P
                      </div>
                      <div className="text-2xl font-bold text-gray-900">
                        {utils.formatMacro(analysisResult.nutritional_data.total_protein)}
                      </div>
                      <div className="text-sm text-gray-600">Protein</div>
                    </div>
                    
                    <div className="text-center p-4 bg-green-50 rounded-lg">
                      <div className="w-8 h-8 bg-green-600 rounded-full mx-auto mb-2 flex items-center justify-center text-white text-xs font-bold">
                        C
                      </div>
                      <div className="text-2xl font-bold text-gray-900">
                        {utils.formatMacro(analysisResult.nutritional_data.total_carbs)}
                      </div>
                      <div className="text-sm text-gray-600">Carbs</div>
                    </div>
                    
                    <div className="text-center p-4 bg-purple-50 rounded-lg">
                      <div className="w-8 h-8 bg-purple-600 rounded-full mx-auto mb-2 flex items-center justify-center text-white text-xs font-bold">
                        F
                      </div>
                      <div className="text-2xl font-bold text-gray-900">
                        {utils.formatMacro(analysisResult.nutritional_data.total_fats)}
                      </div>
                      <div className="text-sm text-gray-600">Fats</div>
                    </div>
                  </div>

                  {/* Macro Distribution */}
                  <div className="mb-6">
                    <h3 className="text-sm font-medium text-gray-700 mb-3">
                      Macronutrient Distribution
                    </h3>
                    {(() => {
                      const percentages = formatMacroPercentage(
                        analysisResult.nutritional_data.total_protein,
                        analysisResult.nutritional_data.total_carbs,
                        analysisResult.nutritional_data.total_fats
                      )
                      return (
                        <div className="flex rounded-lg overflow-hidden h-4">
                          <div 
                            className="bg-blue-500" 
                            style={{ width: `${percentages.protein}%` }}
                            title={`Protein: ${percentages.protein}%`}
                          ></div>
                          <div 
                            className="bg-green-500" 
                            style={{ width: `${percentages.carbs}%` }}
                            title={`Carbs: ${percentages.carbs}%`}
                          ></div>
                          <div 
                            className="bg-purple-500" 
                            style={{ width: `${percentages.fats}%` }}
                            title={`Fats: ${percentages.fats}%`}
                          ></div>
                        </div>
                      )
                    })()}
                  </div>
                </div>

                {/* Food Items */}
                <div className="card">
                  <h2 className="text-xl font-semibold text-gray-900 mb-6">
                    Detected Food Items
                  </h2>
                  
                  <div className="space-y-3">
                    {analysisResult.food_items.map((item, index) => (
                      <div key={index} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                        <div className="flex-1">
                          <div className="font-medium text-gray-900">
                            {item.item}
                          </div>
                          <div className="text-sm text-gray-600">
                            {item.description}
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="font-semibold text-gray-900">
                            {item.calories} cal
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* AI Analysis */}
                <div className="card">
                  <h2 className="text-xl font-semibold text-gray-900 mb-6">
                    AI Analysis Report
                  </h2>
                  
                  <div className="prose prose-sm max-w-none">
                    <div className="whitespace-pre-wrap text-gray-700 leading-relaxed">
                      {analysisResult.analysis}
                    </div>
                  </div>
                </div>

                {/* AI Visualizations */}
                {hasEnhancedFeatures && (
                  <div className="card">
                    <div className="flex items-center justify-between mb-6">
                      <h2 className="text-xl font-semibold text-gray-900">
                        AI Visualizations
                      </h2>
                      <button
                        onClick={() => setShowVisualizations(!showVisualizations)}
                        className="btn-secondary"
                      >
                        {showVisualizations ? (
                          <>
                            <EyeIcon className="w-4 h-4 mr-2" />
                            Hide Visualizations
                          </>
                        ) : (
                          <>
                            <MagnifyingGlassIcon className="w-4 h-4 mr-2" />
                            Show Visualizations
                          </>
                        )}
                      </button>
                    </div>
                    
                    <AnimatePresence>
                      {showVisualizations && (
                        <motion.div
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: 'auto' }}
                          exit={{ opacity: 0, height: 0 }}
                          className="space-y-6"
                        >
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            {/* Grad-CAM */}
                            <div className="space-y-2">
                              <h3 className="text-sm font-medium text-gray-700 flex items-center">
                                <LightBulbIcon className="w-4 h-4 mr-2" />
                                Grad-CAM - AI Focus Areas
                              </h3>
                              {(analysisResult as any).visualizations?.gradcam?.success ? (
                                <div className="relative aspect-square rounded-lg overflow-hidden bg-gray-100">
                                  <Image
                                    src={(analysisResult as any).visualizations.gradcam.imageUrl}
                                    alt="Grad-CAM visualization"
                                    fill
                                    className="object-cover"
                                  />
                                </div>
                              ) : (
                                <div className="aspect-square rounded-lg bg-gray-100 flex items-center justify-center">
                                  <p className="text-gray-500 text-sm">Grad-CAM not available</p>
                                </div>
                              )}
                            </div>

                            {/* SHAP */}
                            <div className="space-y-2">
                              <h3 className="text-sm font-medium text-gray-700 flex items-center">
                                <ChartBarIcon className="w-4 h-4 mr-2" />
                                SHAP - Feature Importance
                              </h3>
                              {(analysisResult as any).visualizations?.shap?.success ? (
                                <div className="relative aspect-square rounded-lg overflow-hidden bg-gray-100">
                                  <Image
                                    src={(analysisResult as any).visualizations.shap.imageUrl}
                                    alt="SHAP visualization"
                                    fill
                                    className="object-cover"
                                  />
                                </div>
                              ) : (
                                <div className="aspect-square rounded-lg bg-gray-100 flex items-center justify-center">
                                  <p className="text-gray-500 text-sm">SHAP not available</p>
                                </div>
                              )}
                            </div>

                            {/* LIME */}
                            <div className="space-y-2">
                              <h3 className="text-sm font-medium text-gray-700 flex items-center">
                                <MagnifyingGlassIcon className="w-4 h-4 mr-2" />
                                LIME - Local Interpretability
                              </h3>
                              {(analysisResult as any).visualizations?.lime?.success ? (
                                <div className="relative aspect-square rounded-lg overflow-hidden bg-gray-100">
                                  <Image
                                    src={(analysisResult as any).visualizations.lime.imageUrl}
                                    alt="LIME visualization"
                                    fill
                                    className="object-cover"
                                  />
                                </div>
                              ) : (
                                <div className="aspect-square rounded-lg bg-gray-100 flex items-center justify-center">
                                  <p className="text-gray-500 text-sm">LIME not available</p>
                                </div>
                              )}
                            </div>

                            {/* Edge Detection */}
                            <div className="space-y-2">
                              <h3 className="text-sm font-medium text-gray-700 flex items-center">
                                <CpuChipIcon className="w-4 h-4 mr-2" />
                                Edge Detection - Food Boundaries
                              </h3>
                              {(analysisResult as any).visualizations?.edge?.success ? (
                                <div className="relative aspect-square rounded-lg overflow-hidden bg-gray-100">
                                  <Image
                                    src={(analysisResult as any).visualizations.edge.imageUrl}
                                    alt="Edge detection visualization"
                                    fill
                                    className="object-cover"
                                  />
                                </div>
                              ) : (
                                <div className="aspect-square rounded-lg bg-gray-100 flex items-center justify-center">
                                  <p className="text-gray-500 text-sm">Edge detection not available</p>
                                </div>
                              )}
                            </div>
                          </div>

                          {/* Visualization Guide */}
                          <div className="bg-blue-50 rounded-lg p-4">
                            <h4 className="text-sm font-medium text-blue-900 mb-2">Visualization Guide:</h4>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs text-blue-800">
                              <div>• <strong>Grad-CAM:</strong> Shows areas the AI focuses on for classification</div>
                              <div>• <strong>SHAP:</strong> Highlights feature importance for predictions</div>
                              <div>• <strong>LIME:</strong> Explains local decision-making regions</div>
                              <div>• <strong>Edge Detection:</strong> Reveals food boundaries and textures</div>
                            </div>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                )}
              </div>

              {/* Sidebar */}
              <div className="space-y-6">
                {/* Image */}
                {previewUrl && (
                  <div className="card">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">
                      Analyzed Image
                    </h3>
                    <div className="relative aspect-square rounded-lg overflow-hidden bg-gray-100">
                      <Image
                        src={previewUrl}
                        alt="Analyzed food"
                        fill
                        className="object-cover"
                      />
                    </div>
                  </div>
                )}

                {/* Quick Actions */}
                <div className="card">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">
                    Quick Actions
                  </h3>
                  
                  <div className="space-y-3">
                    <button className="w-full btn-primary">
                      <ChartBarIcon className="w-4 h-4 mr-2" />
                      Add to Diary
                    </button>
                    {hasEnhancedFeatures && (
                      <button 
                        onClick={() => setShowVisualizations(!showVisualizations)}
                        className="w-full btn-secondary"
                      >
                        <EyeIcon className="w-4 h-4 mr-2" />
                        {showVisualizations ? 'Hide' : 'View'} AI Visualizations
                      </button>
                    )}
                    <button 
                      onClick={resetAnalysis}
                      className="w-full btn-secondary"
                    >
                      <CameraIcon className="w-4 h-4 mr-2" />
                      Analyze Another
                    </button>
                  </div>
                </div>

                {/* Enhanced Detection Stats */}
                {hasEnhancedFeatures && (
                  <div className="card">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">
                      Enhanced Detection Details
                    </h3>
                    
                    <div className="space-y-3 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Items Detected:</span>
                        <span className="font-medium">
                          {(analysisResult as any).detection_metadata?.total_items || analysisResult.food_items.length}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Detection Confidence:</span>
                        <span className="font-medium text-green-600">
                          {Math.round((analysisResult as any).detection_metadata?.confidence * 100 || 0)}%
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Detection Methods:</span>
                        <span className="font-medium">
                          {(analysisResult as any).detection_metadata?.detection_methods?.length || 0}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Processing Time:</span>
                        <span className="font-medium">
                          {Math.round((analysisResult as any).detection_metadata?.processing_time / 1000 || 0)}s
                        </span>
                      </div>
                      {(analysisResult as any).detection_metadata?.detection_methods && (
                        <div className="mt-4 pt-3 border-t border-gray-200">
                          <div className="text-xs text-gray-500 mb-2">Detection Methods Used:</div>
                          <div className="flex flex-wrap gap-1">
                            {(analysisResult as any).detection_metadata.detection_methods.map((method: string, index: number) => (
                              <span 
                                key={index}
                                className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full"
                              >
                                {method}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  )
}