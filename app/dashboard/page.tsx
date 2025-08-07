'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  CameraIcon, 
  ChartBarIcon, 
  ClockIcon, 
  FireIcon,
  EyeIcon,
  BeakerIcon,
  ArrowRightIcon,
  SparklesIcon,
  TrophyIcon,
  CalendarIcon
} from '@heroicons/react/24/outline'
import Link from 'next/link'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'
import React from 'react'

// Mock data for charts
const weeklyData = [
  { day: 'Mon', calories: 2100, protein: 120, carbs: 250, fats: 70 },
  { day: 'Tue', calories: 1950, protein: 110, carbs: 220, fats: 65 },
  { day: 'Wed', calories: 2200, protein: 130, carbs: 280, fats: 75 },
  { day: 'Thu', calories: 2050, protein: 115, carbs: 240, fats: 68 },
  { day: 'Fri', calories: 2300, protein: 140, carbs: 290, fats: 80 },
  { day: 'Sat', calories: 2400, protein: 145, carbs: 300, fats: 85 },
  { day: 'Sun', calories: 2150, protein: 125, carbs: 260, fats: 72 }
]

const macroData = [
  { name: 'Protein', value: 25, color: '#3b82f6' },
  { name: 'Carbs', value: 50, color: '#10b981' },
  { name: 'Fats', value: 25, color: '#f59e0b' }
]

const recentAnalyses = [
  { id: 1, food: 'Grilled Chicken Salad', calories: 450, time: '2 hours ago', accuracy: 95 },
  { id: 2, food: 'Pasta with Marinara', calories: 680, time: '5 hours ago', accuracy: 92 },
  { id: 3, food: 'Avocado Toast', calories: 320, time: '1 day ago', accuracy: 98 },
  { id: 4, food: 'Protein Smoothie', calories: 280, time: '1 day ago', accuracy: 90 }
]

const quickStats = [
  { 
    label: 'Today\'s Calories', 
    value: '1,850', 
    target: '2,000',
    icon: FireIcon, 
    color: 'text-red-600',
    bgColor: 'bg-red-50',
    progress: 92.5
  },
  { 
    label: 'Meals Analyzed', 
    value: '127', 
    change: '+12',
    icon: CameraIcon, 
    color: 'text-blue-600',
    bgColor: 'bg-blue-50'
  },
  { 
    label: 'Avg Accuracy', 
    value: '94.2%', 
    change: '+2.1%',
    icon: EyeIcon, 
    color: 'text-green-600',
    bgColor: 'bg-green-50'
  },
  { 
    label: 'Streak Days', 
    value: '15', 
    change: '+1',
    icon: TrophyIcon, 
    color: 'text-purple-600',
    bgColor: 'bg-purple-50'
  }
]

export default function DashboardPage() {
  const [mounted, setMounted] = useState(false)
  const [currentTime, setCurrentTime] = useState(new Date())

  useEffect(() => {
    setMounted(true)
    const timer = setInterval(() => setCurrentTime(new Date()), 1000)
    return () => clearInterval(timer)
  }, [])

  if (!mounted) {
    return null
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-gray-100">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-gray-200 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-4">
              <Link href="/" className="flex items-center space-x-2">
                <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
                  <SparklesIcon className="w-5 h-5 text-white" />
                </div>
                <span className="text-xl font-bold text-gray-900">AI Calorie App</span>
              </Link>
              <div className="hidden md:block text-sm text-gray-500">
                {currentTime.toLocaleDateString('en-US', { 
                  weekday: 'long', 
                  year: 'numeric', 
                  month: 'long', 
                  day: 'numeric' 
                })}
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <Link href="/analyze" className="btn-primary">
                <CameraIcon className="w-5 h-5 mr-2" />
                Analyze Food
              </Link>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Welcome Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-2">
            Welcome back! 👋
          </h1>
          <p className="text-lg text-gray-600">
            Here's your nutrition overview and AI analysis dashboard.
          </p>
        </motion.div>

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
                <div className={`p-3 rounded-lg ${stat.bgColor}`}>
                  <stat.icon className={`w-6 h-6 ${stat.color}`} />
                </div>
                {stat.change && (
                  <span className="text-sm font-medium text-green-600">
                    {stat.change}
                  </span>
                )}
              </div>
              <div className="space-y-1">
                <p className="text-2xl font-bold text-gray-900">{stat.value}</p>
                <p className="text-sm text-gray-600">{stat.label}</p>
                {stat.target && (
                  <div className="mt-2">
                    <div className="flex justify-between text-xs text-gray-500 mb-1">
                      <span>Progress</span>
                      <span>{stat.progress}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-primary-600 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${stat.progress}%` }}
                      />
                    </div>
                  </div>
                )}
              </div>
            </motion.div>
          ))}
        </div>

        {/* Main Content Grid */}
        <div className="grid lg:grid-cols-3 gap-8">
          {/* Charts Section */}
          <div className="lg:col-span-2 space-y-8">
            {/* Weekly Calories Chart */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="card"
            >
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-xl font-semibold text-gray-900">Weekly Overview</h3>
                <div className="flex items-center space-x-2 text-sm text-gray-500">
                  <CalendarIcon className="w-4 h-4" />
                  <span>Last 7 days</span>
                </div>
              </div>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={weeklyData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey="day" stroke="#6b7280" />
                    <YAxis stroke="#6b7280" />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'white', 
                        border: '1px solid #e5e7eb',
                        borderRadius: '8px',
                        boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                      }}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="calories" 
                      stroke="#22c55e" 
                      strokeWidth={3}
                      dot={{ fill: '#22c55e', strokeWidth: 2, r: 4 }}
                      activeDot={{ r: 6, stroke: '#22c55e', strokeWidth: 2 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </motion.div>

            {/* Macro Distribution */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="card"
            >
              <h3 className="text-xl font-semibold text-gray-900 mb-6">Macronutrient Distribution</h3>
              <div className="flex items-center justify-center">
                <div className="h-64 w-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={macroData}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={100}
                        paddingAngle={5}
                        dataKey="value"
                      >
                        {macroData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div className="ml-8 space-y-3">
                  {macroData.map((macro) => (
                    <div key={macro.name} className="flex items-center space-x-3">
                      <div 
                        className="w-4 h-4 rounded-full"
                        style={{ backgroundColor: macro.color }}
                      />
                      <span className="text-gray-700">{macro.name}</span>
                      <span className="font-semibold text-gray-900">{macro.value}%</span>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          </div>

          {/* Sidebar */}
          <div className="space-y-8">
            {/* Quick Actions */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 }}
              className="card"
            >
              <h3 className="text-xl font-semibold text-gray-900 mb-6">Quick Actions</h3>
              <div className="space-y-4">
                <Link 
                  href="/analyze" 
                  className="w-full btn-primary flex items-center justify-center"
                >
                  <CameraIcon className="w-5 h-5 mr-2" />
                  Analyze New Food
                </Link>
                <button className="w-full btn-secondary flex items-center justify-center">
                  <ChartBarIcon className="w-5 h-5 mr-2" />
                  View Reports
                </button>
                <button className="w-full btn-secondary flex items-center justify-center">
                  <BeakerIcon className="w-5 h-5 mr-2" />
                  AI Insights
                </button>
              </div>
            </motion.div>

            {/* Recent Analyses */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.5 }}
              className="card"
            >
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-xl font-semibold text-gray-900">Recent Analyses</h3>
                <ClockIcon className="w-5 h-5 text-gray-400" />
              </div>
              <div className="space-y-4">
                {recentAnalyses.map((analysis) => (
                  <div key={analysis.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors duration-200">
                    <div className="flex-1">
                      <p className="font-medium text-gray-900 text-sm">{analysis.food}</p>
                      <p className="text-xs text-gray-500">{analysis.time}</p>
                    </div>
                    <div className="text-right">
                      <p className="font-semibold text-gray-900 text-sm">{analysis.calories} cal</p>
                      <p className="text-xs text-green-600">{analysis.accuracy}% accuracy</p>
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>

            {/* AI Models Status */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.6 }}
              className="card"
            >
              <h3 className="text-xl font-semibold text-gray-900 mb-6">AI Models Status</h3>
              <div className="space-y-3">
                {[
                  { name: 'BLIP', status: 'Active', color: 'green' },
                  { name: 'YOLO', status: 'Active', color: 'green' },
                  { name: 'CNN', status: 'Active', color: 'green' },
                  { name: 'LLM', status: 'Active', color: 'green' }
                ].map((model) => (
                  <div key={model.name} className="flex items-center justify-between">
                    <span className="text-gray-700 font-medium">{model.name}</span>
                    <div className="flex items-center space-x-2">
                      <div className={`w-2 h-2 rounded-full bg-${model.color}-500`} />
                      <span className={`text-sm text-${model.color}-600`}>{model.status}</span>
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  )
}