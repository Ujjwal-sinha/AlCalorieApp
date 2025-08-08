import { NextRequest, NextResponse } from 'next/server'
import { getApiConfig } from '../../../lib/config'

// BLIP analysis endpoint that forwards requests to Python backend
export async function POST(request: NextRequest) {
  try {
    const apiConfig = getApiConfig()
    const body = await request.json()
    
    const { image, prompt, max_tokens = 200, temperature = 0.3 } = body

    if (!image) {
      return NextResponse.json(
        { success: false, error: 'No image provided' },
        { status: 400 }
      )
    }

    console.log('🔍 Forwarding BLIP analysis request to Python backend...')

    // Forward request to Python BLIP endpoint
    const pythonResponse = await fetch(`${apiConfig.baseUrl}/api/blip-analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image,
        prompt: prompt || 'Describe the food items in this image in detail:',
        max_tokens,
        temperature
      }),
      signal: AbortSignal.timeout(apiConfig.timeout)
    })

    if (!pythonResponse.ok) {
      throw new Error(`Python BLIP API failed: ${pythonResponse.status}`)
    }

    const result = await pythonResponse.json()
    
    console.log('✅ BLIP analysis completed successfully')
    
    return NextResponse.json({
      success: true,
      description: result.description || result.caption || '',
      confidence: result.confidence || 0.8,
      processing_time: result.processing_time || 0
    })

  } catch (error) {
    console.error('❌ BLIP analysis error:', error)
    
    // Return mock response as fallback
    return NextResponse.json({
      success: true,
      description: 'Mock BLIP response: This appears to be a food image containing various ingredients and prepared items.',
      confidence: 0.7,
      fallback: true,
      error: error instanceof Error ? error.message : 'BLIP analysis failed'
    })
  }
}

export async function GET() {
  return NextResponse.json({
    message: 'BLIP Food Analysis API',
    status: 'active',
    version: '1.0.0',
    description: 'Endpoint for BLIP-based food detection and analysis'
  })
}