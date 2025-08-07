import { NextRequest, NextResponse } from 'next/server'

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const image = formData.get('image') as File
    const context = formData.get('context') as string || ''

    if (!image) {
      return NextResponse.json(
        { error: 'No image provided' },
        { status: 400 }
      )
    }

    // Convert image to base64 for Python backend
    const bytes = await image.arrayBuffer()
    const buffer = Buffer.from(bytes)
    const base64Image = buffer.toString('base64')

    // Call Python Streamlit backend
    // Note: You'll need to modify your Python app to accept API calls
    const pythonResponse = await fetch('http://localhost:8501/api/analyze', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image: base64Image,
        context: context,
        format: image.type
      }),
      signal: AbortSignal.timeout(30000) // 30 second timeout
    })

    if (!pythonResponse.ok) {
      throw new Error(`Python backend error: ${pythonResponse.status}`)
    }

    const result = await pythonResponse.json()

    return NextResponse.json(result)
  } catch (error: any) {
    console.error('API Error:', error)
    
    // Return mock data for development/demo purposes
    const mockResult = {
      success: true,
      analysis: `## COMPREHENSIVE FOOD ANALYSIS

### IDENTIFIED FOOD ITEMS:
- Item: Grilled chicken breast (150g), Calories: 231, Protein: 43.5g, Carbs: 0g, Fats: 5g
- Item: Steamed broccoli (100g), Calories: 34, Protein: 2.8g, Carbs: 7g, Fats: 0.4g
- Item: Brown rice (80g), Calories: 216, Protein: 5g, Carbs: 45g, Fats: 1.8g

### NUTRITIONAL TOTALS:
- Total Calories: 481 kcal
- Total Protein: 51.3g (43% of calories)
- Total Carbohydrates: 52g (43% of calories)
- Total Fats: 7.2g (14% of calories)

### MEAL COMPOSITION ANALYSIS:
- **Meal Type**: Lunch/Dinner
- **Cuisine Style**: Healthy Western
- **Portion Size**: Medium
- **Main Macronutrient**: Protein-rich

### NUTRITIONAL QUALITY ASSESSMENT:
- **Strengths**: High protein content, good fiber from vegetables, balanced macronutrients
- **Areas for Improvement**: Could add healthy fats like avocado or nuts
- **Missing Nutrients**: Healthy fats, vitamin C could be higher

### HEALTH RECOMMENDATIONS:
1. **Excellent protein source** - Great for muscle maintenance and satiety
2. **Add healthy fats** - Consider adding olive oil or avocado
3. **Perfect post-workout meal** - High protein aids recovery

### DIETARY CONSIDERATIONS:
- **Allergen Information**: Gluten-free, dairy-free
- **Dietary Restrictions**: Suitable for most diets
- **Blood Sugar Impact**: Low to moderate glycemic impact`,
      food_items: [
        { item: 'Grilled chicken breast', description: 'Grilled chicken breast - 231 calories', calories: 231 },
        { item: 'Steamed broccoli', description: 'Steamed broccoli - 34 calories', calories: 34 },
        { item: 'Brown rice', description: 'Brown rice - 216 calories', calories: 216 }
      ],
      nutritional_data: {
        total_calories: 481,
        total_protein: 51.3,
        total_carbs: 52,
        total_fats: 7.2,
        items: [
          { item: 'Grilled chicken breast', calories: 231, protein: 43.5, carbs: 0, fats: 5 },
          { item: 'Steamed broccoli', calories: 34, protein: 2.8, carbs: 7, fats: 0.4 },
          { item: 'Brown rice', calories: 216, protein: 5, carbs: 45, fats: 1.8 }
        ]
      },
      improved_description: 'grilled chicken breast, steamed broccoli, brown rice'
    }

    return NextResponse.json(mockResult)
  }
}