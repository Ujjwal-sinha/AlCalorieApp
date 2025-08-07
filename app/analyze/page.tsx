'use client'

import { useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useDropzone } from 'react-dropzone'
import { 
  CameraIcon, 
  CloudArrowUpIcon, 
  SparklesIcon,
  EyeIcon,
  BeakerIcon,
  ChartBarIcon,
  ArrowLeftIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline'
import Link from 'next/link'
import Image from 'next/image'
import axios from 'axios'
import React from 'react'

interface NutritionData {
  total_calories: number
  total_protein: number
  total_carbs: number
  total_fats: number
  items: Array<{
    item: string
    calories: number
    protein: number
    carbs: number
    fats: number
  }>
}

interface AnalysisResult {
  success: boolean
  analysis: string
  food_items: Array<{
    item: string
    description: string
    calories: number
  }>
  nutritional_data: NutritionData
  improved_description: string
}

export default function AnalyzePage() {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null)
  const [context, setContext] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [analysisStep, setAnalysisStep] = useState(0)

  const analysisSteps = [
    { label: 'Loading image...', progress: 10 },
    { label: 'Detecting food items...', progress: 30 },
    { label: 'Analyzing nutrition...', progress: 70 },
    { label: 'Analysis complete!', progress: 100 }
  ]

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = () => {
        setUploadedImage(reader.result as string)
        setAnalysisResult(null)
        setError(null)
      }
      reader.readAsDataURL(file)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.webp']
    },
    multiple: false,
    maxSize: 10 * 1024 * 1024 // 10MB
  })

  const analyzeFood = async () => {
    if (!uploadedImage) return

    setIsAnalyzing(true)
    setError(null)
    setAnalysisStep(0)

    try {
      // Simulate step progression
      const stepInterval = setInterval(() => {
        setAnalysisStep(prev => {
          if (prev < analysisSteps.length - 1) {
            return prev + 1
          }
          clearInterval(stepInterval)
          return prev
        })
      }, 2000)

      // Convert base64 to blob
      const response = await fetch(uploadedImage)
      const blob = await response.blob()
      
      // Create FormData
      const formData = new FormData()
      formData.append('image', blob, 'food-image.jpg')
      formData.append('context', context)

      // Call Python backend (you'll need to implement this endpoint)
      const analysisResponse = await axios.post('/api/analyze-food', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 30000 // 30 seconds timeout
      })

      clearInterval(stepInterval)
      setAnalysisStep(analysisSteps.length - 1)
      setAnalysisResult(analysisResponse.data)
    } catch (err: any) {
      setError(err.response?.data?.error || 'Analysis failed. Please try again.')
      console.error('Analysis error:', err)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const resetAnalysis = () => {
    setUploadedImage(null)
    setAnalysisResult(null)
    setContext('')
    setError(null)
    setAnalysisStep(0)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-gray-100">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-gray-200 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-4">
              <Link href="/dashboard" className="flex items-center space-x-2 text-gray-600 hover:text-gray-900 transition-colors">
                <ArrowLeftIcon className="w-5 h-5" />
                <span>Back to Dashboard</span>
              </Link>
            </div>
            
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
                <SparklesIcon className="w-5 h-5 text-white" />
              </div>
              <span className="text-xl font-bold text-gray-900">AI Food Analyzer</span>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
            AI-Powered Food Analysis
          </h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Upload a photo of your meal and get comprehensive nutritional analysis 
            powered by advanced AI models including BLIP, YOLO, and machine learning.
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="space-y-6"
          >
            {/* Image Upload */}
            <div className="card">
              <h3 className="text-xl font-semibold text-gray-900 mb-4">Upload Food Image</h3>
              
              {!uploadedImage ? (
                <div
                  {...getRootProps()}
                  className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all duration-200 ${
                    isDragActive 
                      ? 'border-primary-500 bg-primary-50' 
                      : 'border-gray-300 hover:border-primary-400 hover:bg-gray-50'
                  }`}
                >
                  <input {...getInputProps()} />
                  <CloudArrowUpIcon className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-lg font-medium text-gray-900 mb-2">
                    {isDragActive ? 'Drop your image here' : 'Drag & drop your food image'}
                  </p>
                  <p className="text-gray-600 mb-4">
                    or click to browse files
                  </p>
                  <p className="text-sm text-gray-500">
                    Supports JPG, PNG, WebP up to 10MB
                  </p>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="relative rounded-lg overflow-hidden">
                    <img 
                      src={uploadedImage} 
                      alt="Uploaded food" 
                      className="w-full h-64 object-cover"
                    />
                    <button
                      onClick={resetAnalysis}
                      className="absolute top-2 right-2 bg-red-500 text-white p-2 rounded-full hover:bg-red-600 transition-colors"
                    >
                      ×
                    </button>
                  </div>
                  
                  {/* Context Input */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Additional Context (Optional)
                    </label>
                    <textarea
                      value={context}
                      onChange={(e) => setContext(e.target.value)}
                      placeholder="Describe the meal if needed (e.g., 'chicken curry with rice')"
                      className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none"
                      rows={3}
                    />
                  </div>

                  {/* Analyze Button */}
                  <button
                    onClick={analyzeFood}
                    disabled={isAnalyzing}
                    className="w-full btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isAnalyzing ? (
                      <div className="flex items-center justify-center">
                        <div className="loading-dots mr-2">
                          <div></div>
                          <div></div>
                          <div></div>
                          <div></div>
                        </div>
                        Analyzing...
                      </div>
                    ) : (
                      <>
                        <BeakerIcon className="w-5 h-5 mr-2" />
                        Analyze Food
                      </>
                    )}
                  </button>
                </div>
              )}
            </div>

            {/* Analysis Progress */}
            <AnimatePresence>
              {isAnalyzing && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="card"
                >
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Analysis Progress</h3>
                  <div className="space-y-4">
                    {analysisSteps.map((step, index) => (
                      <div key={index} className="flex items-center space-x-3">
                        <div className={`w-6 h-6 rounded-full flex items-center justify-center ${
                          index <= analysisStep 
                            ? 'bg-primary-600 text-white' 
                            : 'bg-gray-200 text-gray-400'
                        }`}>
                          {index < analysisStep ? (
                            <CheckCircleIcon className="w-4 h-4" />
                          ) : index === analysisStep ? (
                            <div className="w-3 h-3 bg-white rounded-full animate-pulse" />
                          ) : (
                            <span className="text-xs">{index + 1}</span>
                          )}
                        </div>
                        <span className={`${
                          index <= analysisStep ? 'text-gray-900' : 'text-gray-500'
                        }`}>
                          {step.label}
                        </span>
                      </div>
                    ))}
                    
                    <div className="mt-4">
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-primary-600 h-2 rounded-full transition-all duration-500"
                          style={{ width: `${analysisSteps[analysisStep]?.progress || 0}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>

          {/* Results Section */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="space-y-6"
          >
            {error && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="card border-red-200 bg-red-50"
              >
                <div className="flex items-center space-x-3">
                  <ExclamationTriangleIcon className="w-6 h-6 text-red-600" />
                  <div>
                    <h3 className="text-lg font-semibold text-red-900">Analysis Failed</h3>
                    <p className="text-red-700">{error}</p>
                  </div>
                </div>
              </motion.div>
            )}

            {analysisResult && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-6"
              >
                {/* Nutrition Summary */}
                <div className="card">
                  <h3 className="text-xl font-semibold text-gray-900 mb-4">Nutrition Summary</h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center p-4 bg-red-50 rounded-lg">
                      <div className="text-2xl font-bold text-red-600">
                        {analysisResult.nutritional_data.total_calories}
                      </div>
                      <div className="text-sm text-gray-600">Calories</div>
                    </div>
                    <div className="text-center p-4 bg-blue-50 rounded-lg">
                      <div className="text-2xl font-bold text-blue-600">
                        {analysisResult.nutritional_data.total_protein.toFixed(1)}g
                      </div>
                      <div className="text-sm text-gray-600">Protein</div>
                    </div>
                    <div className="text-center p-4 bg-green-50 rounded-lg">
                      <div className="text-2xl font-bold text-green-600">
                        {analysisResult.nutritional_data.total_carbs.toFixed(1)}g
                      </div>
                      <div className="text-sm text-gray-600">Carbs</div>
                    </div>
                    <div className="text-center p-4 bg-yellow-50 rounded-lg">
                      <div className="text-2xl font-bold text-yellow-600">
                        {analysisResult.nutritional_data.total_fats.toFixed(1)}g
                      </div>
                      <div className="text-sm text-gray-600">Fats</div>
                    </div>
                  </div>
                </div>

                {/* Detected Food Items */}
                <div className="card">
                  <h3 className="text-xl font-semibold text-gray-900 mb-4">Detected Food Items</h3>
                  <p className="text-gray-600 mb-4">
                    <strong>Foods found:</strong> {analysisResult.improved_description}
                  </p>
                  
                  {analysisResult.food_items.length > 0 && (
                    <div className="space-y-3">
                      {analysisResult.food_items.map((item, index) => (
                        <div key={index} className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                          <span className="font-medium text-gray-900">{item.item}</span>
                          <span className="text-primary-600 font-semibold">{item.calories} cal</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Detailed Analysis */}
                <div className="card">
                  <h3 className="text-xl font-semibold text-gray-900 mb-4">Comprehensive Analysis</h3>
                  <div className="prose prose-sm max-w-none">
                    <div className="whitespace-pre-wrap text-gray-700">
                      {analysisResult.analysis}
                    </div>
                  </div>
                </div>

                {/* AI Visualizations Placeholder */}
                <div className="card">
                  <h3 className="text-xl font-semibold text-gray-900 mb-4">AI Visualizations</h3>
                  <div className="grid grid-cols-2 gap-4">
                    {['Edge Detection', 'Grad-CAM', 'SHAP Analysis', 'LIME Explanation'].map((viz) => (
                      <div key={viz} className="aspect-square bg-gray-100 rounded-lg flex items-center justify-center">
                        <div className="text-center">
                          <EyeIcon className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                          <p className="text-sm text-gray-500">{viz}</p>
                          <p className="text-xs text-gray-400">Coming soon</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </motion.div>
            )}

            {!uploadedImage && !isAnalyzing && !analysisResult && (
              <div className="card text-center">
                <CameraIcon className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-gray-900 mb-2">Ready to Analyze</h3>
                <p className="text-gray-600">
                  Upload a food image to get started with AI-powered nutritional analysis.
                </p>
              </div>
            )}
          </motion.div>
        </div>
      </div>
    </div>
  )
}