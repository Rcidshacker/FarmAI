"""
Economic Threshold Mapper
Translates mealybug population to economic decisions
Research: >8 mealybugs/fruit = intervention threshold
"""
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class EconomicThresholdMapper:
    """
    Map mealybug risk scores to actionable economic thresholds.
    Research basis:
    - Peak: ~105 mealybugs/fruit (untreated)
    - With treatment: ~7-8 mealybugs/fruit (acceptable)
    - Economic threshold: >8 bugs/fruit
    - Yield loss: 30-60% during epidemics
    """
    
    # Thresholds
    MONITORING_THRESHOLD = 5      # Start watching closely
    PREPARE_THRESHOLD = 8         # Get biocontrol ready
    INTERVENE_THRESHOLD = 15      # Spray NOW
    EMERGENCY_THRESHOLD = 30      # Multiple treatments
    
    def __init__(self):
        pass
        
    def map_risk_to_population(self, risk_score: float) -> float:
        """
        Convert risk score (0-1) to estimated mealybugs per fruit.
        Based on research: peak = 105 bugs/fruit at 100% risk
        """
        return risk_score * 105.0
    
    def estimate_yield_loss(self, mealybugs_per_fruit: float) -> float:
        """
        Estimate yield loss percentage.
        Research: ~0.6% loss per mealybug (up to 60% max)
        """
        return min(60, mealybugs_per_fruit * 0.6)
    
    def get_quality_impact(self, mealybugs_per_fruit: float, humidity: float) -> str:
        """
        Assess fruit quality impact.
        Sooty mold = unmarketable (if >15 bugs + high humidity)
        """
        if mealybugs_per_fruit > 15 and humidity > 80:
            return "UNMARKETABLE (Sooty mold expected)"
        elif mealybugs_per_fruit > 10:
            return "Reduced (Honeydew present)"
        elif mealybugs_per_fruit > 5:
            return "Minor cosmetic damage"
        else:
            return "Good"
    
    def get_action_recommendation(self, 
                                  risk_score: float,
                                  humidity: float = 50,
                                  temp: float = 25) -> Dict:
        """
        Provide clear action recommendation based on thresholds.
        
        Returns:
            {
                'action_level': str,
                'estimated_bugs_per_fruit': float,
                'estimated_yield_loss_pct': float,
                'fruit_quality': str,
                'economic_threshold_exceeded': bool,
                'recommended_action': str,
                'urgency': str  # 'low', 'medium', 'high', 'critical'
            }
        """
        bugs_per_fruit = self.map_risk_to_population(risk_score)
        yield_loss = self.estimate_yield_loss(bugs_per_fruit)
        quality = self.get_quality_impact(bugs_per_fruit, humidity)
        
        # Determine action level
        if bugs_per_fruit < self.MONITORING_THRESHOLD:
            action_level = "MONITORING"
            recommended_action = "Weekly scouting, no treatment needed"
            urgency = "low"
            
        elif bugs_per_fruit < self.PREPARE_THRESHOLD:
            action_level = "PREPARE"
            recommended_action = "Prepare biocontrol (order Cryptolaemus, check stock)"
            urgency = "medium"
            
        elif bugs_per_fruit < self.INTERVENE_THRESHOLD:
            action_level = "INTERVENE"
            recommended_action = "Deploy biocontrol NOW or spray botanicals (Azadirachtin)"
            urgency = "high"
            
        elif bugs_per_fruit < self.EMERGENCY_THRESHOLD:
            action_level = "CRITICAL"
            recommended_action = "Chemical intervention (Fenobucarb) + biocontrol"
            urgency = "critical"
            
        else:
            action_level = "EMERGENCY"
            recommended_action = "🚨 Multiple treatments + ant control + pruning"
            urgency = "critical"
        
        return {
            'action_level': action_level,
            'estimated_bugs_per_fruit': round(bugs_per_fruit, 1),
            'estimated_yield_loss_pct': round(yield_loss, 1),
            'fruit_quality': quality,
            'economic_threshold_exceeded': bugs_per_fruit > self.PREPARE_THRESHOLD,
            'recommended_action': recommended_action,
            'urgency': urgency
        }
