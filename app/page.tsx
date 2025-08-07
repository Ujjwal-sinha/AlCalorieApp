'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  CameraIcon, 
  ChartBarIcon, 
  SparklesIcon, 
  ArrowRightIcon,
  EyeIcon,
  BeakerIcon,
  CpuChipIcon,
  LightBulbIcon
} from '@heroicons/react/24/outline'
import Link from 'next/link'
import Image from 'next/image'
import React from 'react'

const features = [
  {
    icon: CameraIcon,
    title: 'AI-Powered Detection',
    description: 'Advanced computer vision using BLIP and YOLO models to identify every food item in your image.',
    color: 'text-blue-600'
  },
  {
    icon: ChartBarIcon,
    title: 'Comprehensive Analysis',
    description: 'Detailed nutritional breakdown with calories, macronutrients, and health recommendations.',
    color: 'text-green-600'
  },
  {
    icon: SparklesIcon,
    title: 'AI Visualizations',
    description: 'See how AI thinks with Grad-CAM, SHAP, LIME, and edge detection visualizations.',
    color: 'text-purple-600'
  },
  {
    icon: BeakerIcon,
    title: 'Multi-Model Approach',
    description: 'Combines multiple AI models for maximum accuracy and comprehensive food detection.',
    color: 'text-orange-600'
  }
]

const stats = [
  { label: 'Food Items Detected', value: '10,000+', icon: CameraIcon },
  { label: 'Accuracy Rate', value: '95%+', icon: EyeIcon },
  { label: 'AI Models Used', value: '4+', icon: CpuChipIcon },
  { label: 'Analysis Speed', value: '<15s', icon: LightBulbIcon }
]

export default function HomePage() {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) {
    return null
  }

  return (
    <div className="min-h-screen">
      {/* Navigation */}
      <nav className="fixed top-0 w-full z-50 glass-effect border-b border-white/20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <motion.div 
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="flex items-center space-x-2"
            >
              <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
                <SparklesIcon className="w-5 h-5 text-white" />
              </div>
              <span className="text-xl font-bold text-gray-900">AI Calorie App</span>
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
            >
              <Link href="/dashboard" className="btn-primary">
                Get Started
              </Link>
            </motion.div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-24 pb-16 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="text-center">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
            >
              <h1 className="text-5xl md:text-7xl font-bold text-gray-900 mb-6">
                Smart Food Detection
                <span className="block gradient-bg bg-clip-text text-transparent">
                  Powered by AI
                </span>
              </h1>
              
              <p className="text-xl md:text-2xl text-gray-600 mb-8 max-w-3xl mx-auto leading-relaxed">
                Advanced AI models analyze your food images to provide comprehensive nutritional insights, 
                calorie tracking, and health recommendations with unprecedented accuracy.
              </p>
              
              <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
                <Link href="/dashboard" className="btn-primary text-lg px-8 py-4">
                  Start Analyzing Food
                  <ArrowRightIcon className="w-5 h-5 ml-2 inline" />
                </Link>
                <Link href="/analyze" className="btn-secondary text-lg px-8 py-4">
                  Try Demo
                </Link>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 px-4 sm:px-6 lg:px-8 bg-white/50">
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className="text-center"
              >
                <div className="inline-flex items-center justify-center w-12 h-12 bg-primary-100 rounded-lg mb-4">
                  <stat.icon className="w-6 h-6 text-primary-600" />
                </div>
                <div className="text-3xl font-bold text-gray-900 mb-2">{stat.value}</div>
                <div className="text-gray-600">{stat.label}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
              Advanced AI Technology
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Our cutting-edge AI system combines multiple machine learning models 
              to deliver the most accurate food detection and nutritional analysis available.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                className="card hover:scale-105 transition-transform duration-300"
              >
                <div className={`inline-flex items-center justify-center w-12 h-12 rounded-lg mb-4 ${feature.color} bg-opacity-10`}>
                  <feature.icon className={`w-6 h-6 ${feature.color}`} />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-3">
                  {feature.title}
                </h3>
                <p className="text-gray-600 leading-relaxed">
                  {feature.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8 gradient-bg">
        <div className="max-w-4xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
              Ready to Transform Your Nutrition?
            </h2>
            <p className="text-xl text-green-100 mb-8 max-w-2xl mx-auto">
              Join thousands of users who are already using AI to make smarter food choices 
              and achieve their health goals.
            </p>
            <Link href="/dashboard" className="inline-flex items-center bg-white text-primary-600 font-semibold py-4 px-8 rounded-lg hover:bg-gray-50 transition-colors duration-200 shadow-lg hover:shadow-xl">
              Start Your Journey
              <ArrowRightIcon className="w-5 h-5 ml-2" />
            </Link>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="text-center">
            <div className="flex items-center justify-center space-x-2 mb-4">
              <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
                <SparklesIcon className="w-5 h-5 text-white" />
              </div>
              <span className="text-xl font-bold">AI Calorie App</span>
            </div>
            <p className="text-gray-400 mb-6">
              Advanced AI-powered food detection and nutrition analysis
            </p>
            <div className="flex justify-center space-x-6 mb-8">
              <a 
                href="https://github.com/Ujjwal-sinha" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-gray-400 hover:text-white transition-colors duration-200"
              >
                GitHub
              </a>
              <a 
                href="https://www.linkedin.com/in/sinhaujjwal01/" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-gray-400 hover:text-white transition-colors duration-200"
              >
                LinkedIn
              </a>
            </div>
            <div className="border-t border-gray-800 pt-8">
              <p className="text-gray-400">
                © 2024 Ujjwal Sinha • Built with ❤️ using Next.js, TypeScript, and Advanced AI
              </p>
              <p className="text-gray-500 text-sm mt-2">
                🚀 Enhanced Food Detection • 🔬 AI Interpretability • 📊 Nutrition Analysis
              </p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}