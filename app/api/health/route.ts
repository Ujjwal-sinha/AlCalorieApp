import { NextResponse } from 'next/server'
import { getApiConfig } from '../../../lib/config'

export async function GET() {
  try {
    const apiConfig = getApiConfig()
    
    // Check if environment variables are set
    const hasGroqKey = !!process.env.NEXT_PUBLIC_GROQ_API_KEY
    const pythonApiUrl = apiConfig.baseUrl

    // Try to check Python API health if available
    let pythonApiStatus = 'unknown'
    try {
      const response = await fetch(`${pythonApiUrl}/health`, { 
        method: 'GET',
        signal: AbortSignal.timeout(apiConfig.healthCheckTimeout)
      })
      pythonApiStatus = response.ok ? 'active' : 'error'
    } catch (error) {
      pythonApiStatus = 'inactive'
    }

    return NextResponse.json({
      status: 'healthy',
      timestamp: new Date().toISOString(),
      services: {
        nextjs: 'active',
        typescript_analyzer: 'active',
        python_api: pythonApiStatus,
        groq_api: hasGroqKey ? 'configured' : 'not_configured'
      },
      models_available: {
        typescript_mock: true,
        python_blip: pythonApiStatus === 'active',
        python_yolo: pythonApiStatus === 'active',
        python_llm: pythonApiStatus === 'active' && hasGroqKey
      },
      api_version: '1.0.0',
      mode: hasGroqKey ? 'production' : 'mock'
    })
  } catch (error) {
    return NextResponse.json({
      status: 'error',
      error: error instanceof Error ? error.message : 'Unknown error',
      timestamp: new Date().toISOString()
    }, { status: 500 })
  }
}