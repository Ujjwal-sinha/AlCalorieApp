import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import React from 'react'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'AI Calorie App - Smart Food Detection & Nutrition Analysis',
  description: 'Advanced AI-powered food detection and nutrition analysis with comprehensive calorie tracking, powered by BLIP, YOLO, and advanced machine learning models.',
  keywords: 'AI, food detection, calorie tracking, nutrition analysis, machine learning, BLIP, YOLO',
  authors: [{ name: 'Ujjwal Sinha', url: 'https://github.com/Ujjwal-sinha' }],
  openGraph: {
    title: 'AI Calorie App - Smart Food Detection',
    description: 'Advanced AI-powered food detection and nutrition analysis',
    type: 'website',
    locale: 'en_US',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'AI Calorie App - Smart Food Detection',
    description: 'Advanced AI-powered food detection and nutrition analysis',
  },
  viewport: 'width=device-width, initial-scale=1',
  themeColor: '#22c55e',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="scroll-smooth">
      <body className={`${inter.className} antialiased`}>
        <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-gray-100">
          {children}
        </div>
      </body>
    </html>
  )
}