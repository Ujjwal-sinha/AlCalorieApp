import type { HistoryEntry, TrendData } from '../types';

export class HistoryService {
  private static instance: HistoryService;
  private storageKey = 'food-analyzer-history';

  public static getInstance(): HistoryService {
    if (!HistoryService.instance) {
      HistoryService.instance = new HistoryService();
    }
    return HistoryService.instance;
  }

  saveToHistory(entry: HistoryEntry): void {
    try {
      const history = this.getHistory();
      const updatedHistory = [entry, ...history].slice(0, 100); // Keep last 100 entries
      localStorage.setItem(this.storageKey, JSON.stringify(updatedHistory));
    } catch (error) {
      console.error('Failed to save to history:', error);
    }
  }

  getHistory(): HistoryEntry[] {
    try {
      const stored = localStorage.getItem(this.storageKey);
      if (!stored) return [];
      
      const parsed = JSON.parse(stored);
      return parsed.map((entry: any) => ({
        ...entry,
        timestamp: new Date(entry.timestamp)
      }));
    } catch (error) {
      console.error('Failed to load history:', error);
      return [];
    }
  }

  deleteEntry(id: string): void {
    try {
      const history = this.getHistory();
      const filtered = history.filter(entry => entry.id !== id);
      localStorage.setItem(this.storageKey, JSON.stringify(filtered));
    } catch (error) {
      console.error('Failed to delete entry:', error);
    }
  }

  clearHistory(): void {
    try {
      localStorage.removeItem(this.storageKey);
    } catch (error) {
      console.error('Failed to clear history:', error);
    }
  }

  generateTrendData(days: number = 30): TrendData[] {
    const history = this.getHistory();
    const now = new Date();
    const trends: TrendData[] = [];

    for (let i = days - 1; i >= 0; i--) {
      const date = new Date(now);
      date.setDate(date.getDate() - i);
      const dateStr = date.toISOString().split('T')[0];

      // Find entries for this date
      const dayEntries = history.filter(entry => {
        const entryDate = entry.timestamp.toISOString().split('T')[0];
        return entryDate === dateStr && entry.analysis_result.success;
      });

      // Calculate totals for the day
      let totalCalories = 0;
      let totalProtein = 0;
      let totalCarbs = 0;
      let totalFats = 0;

      dayEntries.forEach(entry => {
        const nutrition = entry.analysis_result.nutritional_data;
        if (nutrition) {
          totalCalories += nutrition.total_calories;
          totalProtein += nutrition.total_protein;
          totalCarbs += nutrition.total_carbs;
          totalFats += nutrition.total_fats;
        }
      });

      trends.push({
        date: dateStr,
        calories: totalCalories,
        protein: totalProtein,
        carbs: totalCarbs,
        fats: totalFats
      });
    }

    return trends;
  }

  getStatistics() {
    const history = this.getHistory().filter(entry => entry.analysis_result.success);
    
    if (history.length === 0) {
      return {
        totalAnalyses: 0,
        averageCalories: 0,
        mostCommonFood: 'N/A',
        totalCalories: 0
      };
    }

    let totalCalories = 0;
    const foodCounts: Record<string, number> = {};

    history.forEach(entry => {
      const nutrition = entry.analysis_result.nutritional_data;
      if (nutrition) {
        totalCalories += nutrition.total_calories;

        nutrition.items.forEach(item => {
          foodCounts[item.name] = (foodCounts[item.name] || 0) + 1;
        });
      }
    });

    const mostCommonFood = Object.entries(foodCounts)
      .sort(([,a], [,b]) => b - a)[0]?.[0] || 'N/A';

    return {
      totalAnalyses: history.length,
      averageCalories: Math.round(totalCalories / history.length),
      mostCommonFood,
      totalCalories
    };
  }
}