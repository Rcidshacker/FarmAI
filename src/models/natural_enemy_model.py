"""
Natural Enemy (Biocontrol) Effectiveness Model
Accounts for predators and fungal pathogens that suppress mealybugs
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class NaturalEnemyModel:
    """
    Model effectiveness of biological control agents:
    - Predators (Cryptolaemus montrouzieri, etc.)
    - Fungal pathogens (Metarhizium, Beauveria, Lecanicillium)
    """
    
    def __init__(self):
        self.predator_release_date: Optional[datetime] = None
        self.fungal_spray_date: Optional[datetime] = None
        
    def set_predator_release(self, release_date: datetime):
        """Record when predators (e.g., Cryptolaemus) were released"""
        self.predator_release_date = release_date
        logger.info(f"Predator release recorded: {release_date}")
        
    def set_fungal_application(self, spray_date: datetime):
        """Record when fungal biocontrol (Metarhizium, etc.) was applied"""
        self.fungal_spray_date = spray_date
        logger.info(f"Fungal biocontrol recorded: {spray_date}")
        
    def calculate_predator_suppression(self, 
                                      current_date: datetime,
                                      humidity: float) -> float:
        """
        Calculate mealybug suppression from predators.
        Research: Cryptolaemus correlation r=0.85
        
        Returns: Suppression factor (1.0 = no effect, 0.5 = 50% suppression)
        """
        if not self.predator_release_date:
            return 1.0  # No predators = no suppression
            
        days_since_release = (current_date - self.predator_release_date).days
        
        if days_since_release < 0:
            return 1.0  # Future date?
            
        # Establishment phases
        if days_since_release < 7:
            # Week 1: Predators establishing, minimal impact
            return 0.95
        elif days_since_release < 30:
            # Weeks 2-4: Building up, 30% suppression
            return 0.70
        elif days_since_release < 90:
            # Established: 50% suppression (research: r=0.85)
            return 0.50
        else:
            # Long-term: Slight decline (predators may disperse)
            return 0.60
    
    def calculate_fungal_suppression(self,
                                    current_date: datetime,
                                    humidity: float,
                                    temp: float) -> float:
        """
        Calculate suppression from entomopathogenic fungi (EPF).
        Research: 
        - Metarhizium: 96.67% mortality (lab)
        - Lecanicillium: Reduces to 8.85 bugs/fruit (field)
        - CRITICAL: Requires humidity >70% to work!
        
        Returns: Suppression factor (1.0 = no effect, 0.1 = 90% suppression)
        """
        if not self.fungal_spray_date:
            return 1.0
            
        days_since_spray = (current_date - self.fungal_spray_date).days
        
        if days_since_spray < 0 or days_since_spray > 21:
            return 1.0  # Outside effective window
        
        # Humidity-dependent efficacy
        if humidity < 70:
            efficacy = 0.3  # Poor conditions (fungi need moisture)
        elif humidity > 85:
            efficacy = 0.9  # Optimal! (research: 96.67% mortality)
        else:
            efficacy = 0.6  # Moderate
        
        # Temperature effects
        if temp < 20 or temp > 32:
            efficacy *= 0.7  # Suboptimal temp
        
        # Time decay (fungi remain active for ~3 weeks)
        decay = max(0, 1 - (days_since_spray / 21))
        
        final_suppression = 1 - (efficacy * decay)
        return max(0.1, final_suppression)  # At least 90% max suppression
    
    def calculate_total_biocontrol_effect(self,
                                         current_date: datetime,
                                         humidity: float,
                                         temp: float) -> Dict:
        """
        Combined effect of all natural enemies.
        Effects are MULTIPLICATIVE (agents work together).
        
        Returns:
            {
                'total_suppression_factor': float,  # 1.0 = none, 0.3 = 70% suppression
                'predator_effect': float,
                'fungal_effect': float,
                'recommendation': str
            }
        """
        predator_factor = self.calculate_predator_suppression(current_date, humidity)
        fungal_factor = self.calculate_fungal_suppression(current_date, humidity, temp)
        
        # Multiplicative: Both working together is better than either alone
        total_factor = predator_factor * fungal_factor
        
        # Generate recommendation
        if humidity > 80 and not self.fungal_spray_date:
            recommendation = "💡 Conditions perfect for fungal biocontrol (Metarhizium/Lecanicillium)"
        elif predator_factor > 0.8 and humidity < 70:
            recommendation = "Consider releasing Cryptolaemus (predators work in dry conditions)"
        else:
            recommendation = "Biocontrol active"
        
        return {
            'total_suppression_factor': total_factor,
            'predator_effect': 1 - predator_factor,  # Convert to % suppression
            'fungal_effect': 1 - fungal_factor,
            'recommendation': recommendation
        }
