'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  CameraIcon, 
  ChartBarIcon, 
  FireIcon,
  BeakerIcon,
  EyeIcon,
  ClockIcon,
  TrophyIcon,
  ArrowUpIcon,
  ArrowDownIcon
} from '@heroicons/react/24/outline'
import Link from 'next/link'
import { QuickStat, WeeklyData, RecentAnalysis, ModelStatus } from '../../types'
import { config, getNutritionConfig, getMockDataConfig } from '../../lib/config'

export default function DashboardPage() {
  const [mounted, setMounted] = useState(false)
  const nutritionConfig = getNutritionConfig()
  const mockDataConfig = getMockDataConfig()

  // Mock data for dashboard - using configuration
  const quickStats: QuickStat[] = [
    {
      label: 'Today\'s Calories',
      value: '1,847',
      target: nutritionConfig.defaultCalorieTarget.toString(),
      change: '+12%',
      icon: FireIcon,
      color: 'text-orange-600',
      bgColor: 'bg-orange-50',
      progress: Math.round((1847 / nutritionConfig.defaultCalorieTarget) * 100)
    },
  {
    label: 'Protein',
    value: '89g',
    target: '120g',
    change: '+5%',
    icon: ChartBarIcon,
    color: 'text-blue-600',
    bgColor: 'bg-blue-50',
    progress: 74
  },
  {
    label: 'Foods Analyzed',
    value: '23',
    change: '+8',
    icon: CameraIcon,
    color: 'text-green-600',
    bgColor: 'bg-green-50'
  },
    {
      label: 'Accuracy',
      value: mockDataConfig.confidence,
      change: '+2.1%',
      icon: EyeIcon,
      color: 'text-purple-600',
      bgColor: 'bg-purple-50'
    }
    ]

  const weeklyData: WeeklyData[] = [
  { day: 'Mon', calories: 1950, protein: 85, carbs: 220, fats: 65 },
  { day: 'Tue', calories: 2100, protein: 92, carbs: 240, fats: 70 },
  { day: 'Wed', calories: 1800, protein: 78, carbs: 200, fats: 60 },
  { day: 'Thu', calories: 2200, protein: 98, carbs: 250, fats: 75 },
  { day: 'Fri', calories: 1900, protein: 88, carbs: 210, fats: 68 },
  { day: 'Sat', calories: 2300, protein: 105, carbs: 260, fats: 80 },
  { day: 'Sun', calories: 1847, protein: 89, carbs: 205, fats: 62 }
  ]

  const recentAnalyses: RecentAnalysis[] = [
  { id: 1, food: 'Grilled Chicken Salad', calories: 420, time: '2 hours ago', accuracy: 96 },
  { id: 2, food: 'Quinoa Bowl with Vegetables', calories: 380, time: '4 hours ago', accuracy: 94 },
  { id: 3, food: 'Salmon with Brown Rice', calories: 520, time: '6 hours ago', accuracy: 98 },
  { id: 4, food: 'Greek Yogurt with Berries', calories: 180, time: '8 hours ago', accuracy: 92 }
  ]

  const modelStatus: ModelStatus[] = [
  { name: 'BLIP Vision Model', status: 'Active', color: 'green' },
  { name: 'YOLO Detection', status: 'Active', color: 'green' },
  { name: 'LLM Analysis', status: 'Active', color: 'green' },
  { name: 'CNN Visualization', status: 'Active', color: 'green' }
  ]

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) {
    return null
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Navigation */}
      <nav className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
                <BeakerIcon className="w-5 h-5 text-white" />
              </div>
              <span className="text-xl font-bold text-gray-900">AI Calorie Dashboard</span>
            </div>
            
            <div className="flex items-center space-x-4">
              <Link href="/analyze" className="btn-primary">
                <CameraIcon className="w-4 h-4 mr-2" />
                Analyze Food
              </Link>
              <Link href="/" className="text-gray-600 hover:text-gray-900">
                Home
              </Link>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Welcome back! 👋
          </h1>
          <p className="text-gray-600">
            Here's your nutrition overview and AI analysis insights.
          </p>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {quickStats.map((stat, index) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="card"
            >
              <div className="flex items-center justify-between mb-4">
                <div className={`p-2 rounded-lg ${stat.bgColor}`}>
                  <stat.icon className={`w-6 h-6 ${stat.color}`} />
                </div>
                {stat.change && (
                  <div className={`flex items-center text-sm ${
                    stat.change.startsWith('+') ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {stat.change.startsWith('+') ? (
                      <ArrowUpIcon className="w-4 h-4 mr-1" />
                    ) : (
                      <ArrowDownIcon className="w-4 h-4 mr-1" />
                    )}
                    {stat.change}
                  </div>
                )}
              </div>
              
              <div className="mb-2">
                <div className="text-2xl font-bold text-gray-900 mb-1">
                  {stat.value}
                </div>
                {stat.target && (
                  <div className="text-sm text-gray-500">
                    of {stat.target} target
                  </div>
                )}
              </div>
              
              <div className="text-sm text-gray-600 mb-3">
                {stat.label}
              </div>
              
              {stat.progress && (
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className={`h-2 rounded-full ${stat.color.replace('text-', 'bg-')}`}
                    style={{ width: `${stat.progress}%` }}
                  ></div>
                </div>
              )}
            </motion.div>
          ))}
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Weekly Overview */}
          <div className="lg:col-span-2">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="card"
            >
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-semibold text-gray-900">
                  Weekly Overview
                </h2>
                <div className="flex space-x-2">
                  <div className="flex items-center text-sm text-gray-600">
                    <div className="w-3 h-3 bg-blue-500 rounded-full mr-2"></div>
                    Calories
                  </div>
                  <div className="flex items-center text-sm text-gray-600">
                    <div className="w-3 h-3 bg-green-500 rounded-full mr-2"></div>
                    Protein
                  </div>
                </div>
              </div>
              
              <div className="space-y-4">
                {weeklyData.map((day, index) => (
                  <div key={day.day} className="flex items-center space-x-4">
                    <div className="w-12 text-sm font-medium text-gray-600">
                      {day.day}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-1">
                        <div className="text-sm font-medium text-gray-900">
                          {day.calories} cal
                        </div>
                        <div className="text-sm text-gray-500">
                          {day.protein}g protein
                        </div>
                      </div>
                      <div className="flex space-x-1">
                        <div 
                          className="h-2 bg-blue-500 rounded"
                          style={{ width: `${(day.calories / 2500) * 100}%` }}
                        ></div>
                        <div 
                          className="h-2 bg-green-500 rounded"
                          style={{ width: `${(day.protein / 120) * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Recent Analyses */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="card"
            >
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900">
                  Recent Analyses
                </h3>
                <ClockIcon className="w-5 h-5 text-gray-400" />
              </div>
              
              <div className="space-y-3">
                {recentAnalyses.map((analysis) => (
                  <div key={analysis.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <div className="flex-1">
                      <div className="text-sm font-medium text-gray-900 mb-1">
                        {analysis.food}
                      </div>
                      <div className="text-xs text-gray-500">
                        {analysis.time} • {analysis.calories} cal
                      </div>
                    </div>
                    <div className="text-xs font-medium text-green-600">
                      {analysis.accuracy}%
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>

            {/* AI Model Status */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
              className="card"
            >
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900">
                  AI Model Status
                </h3>
                <BeakerIcon className="w-5 h-5 text-gray-400" />
              </div>
              
              <div className="space-y-3">
                {modelStatus.map((model) => (
                  <div key={model.name} className="flex items-center justify-between">
                    <div className="text-sm text-gray-900">
                      {model.name}
                    </div>
                    <div className="flex items-center">
                      <div className={`w-2 h-2 rounded-full mr-2 ${
                        model.color === 'green' ? 'bg-green-500' :
                        model.color === 'yellow' ? 'bg-yellow-500' : 'bg-red-500'
                      }`}></div>
                      <div className={`text-xs font-medium ${
                        model.color === 'green' ? 'text-green-600' :
                        model.color === 'yellow' ? 'text-yellow-600' : 'text-red-600'
                      }`}>
                        {model.status}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>

            {/* Quick Actions */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.7 }}
              className="card"
            >
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Quick Actions
              </h3>
              
              <div className="space-y-3">
                <Link href="/analyze" className="block w-full btn-primary text-center">
                  <CameraIcon className="w-4 h-4 mr-2 inline" />
                  Analyze New Food
                </Link>
                <button className="w-full btn-secondary text-center">
                  <TrophyIcon className="w-4 h-4 mr-2 inline" />
                  View Goals
                </button>
                <button className="w-full btn-secondary text-center">
                  <ChartBarIcon className="w-4 h-4 mr-2 inline" />
                  Export Data
                </button>
              </div>
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  )
}