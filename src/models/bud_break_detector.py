"""
Bud Break Detection for Custard Apple
Critical timing for ant-mediated mealybug transport (Feb-Mar in India)
"""
import logging
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class BudBreakDetector:
    """
    Detect custard apple bud break stage based on:
    1. Calendar timing (Late Feb - March in India)
    2. Temperature accumulation
    3. Recent rainfall (triggers flushing)
    """
    
    # India-specific timing (Maharashtra/Gujarat)
    BUD_BREAK_START_MONTH = 2  # Late February
    BUD_BREAK_END_MONTH = 3    # March
    BUD_BREAK_START_DAY = 15   # Feb 15
    
    # Temperature thresholds
    MIN_TEMP_FOR_FLUSH = 18  # °C minimum for bud activation
    OPTIMAL_TEMP_RANGE = (20, 28)
    
    def __init__(self):
        self.is_bud_break_active = False
        self.accumulated_warmth_days = 0
        
    def check_bud_break(self, 
                       current_date: datetime,
                       temp_max: float,
                       recent_rainfall: float) -> Dict:
        """
        Determine if bud break stage is active.
        
        Args:
            current_date: Current date
            temp_max: Maximum temperature (°C)
            recent_rainfall: Rainfall in last 7 days (mm)
            
        Returns:
            {
                'is_bud_break': bool,
                'phase': str,  # 'dormant', 'early_bud_break', 'active_flush', 'post_flush'
                'ant_transport_risk': float,  # 0-1, multiplier for soil effect
                'recommendation': str
            }
        """
        month = current_date.month
        day = current_date.day
        
        # Check calendar timing
        in_window = False
        if month == 2 and day >= self.BUD_BREAK_START_DAY:
            in_window = True
        elif month == 3:
            in_window = True
        elif month == 4 and day <= 7:  # Early April still relevant
            in_window = True
        
        # Temperature accumulation
        if temp_max >= self.MIN_TEMP_FOR_FLUSH:
            self.accumulated_warmth_days += 1
        else:
            self.accumulated_warmth_days = max(0, self.accumulated_warmth_days - 0.5)
        
        # Determine phase
        phase = 'dormant'
        ant_transport_risk = 0.5  # Base multiplier
        recommendation = "Monitor tree for signs of bud swell"
        
        if in_window and self.accumulated_warmth_days >= 3:
            # Check if conditions are right
            if recent_rainfall >= 10 and temp_max >= self.MIN_TEMP_FOR_FLUSH:
                phase = 'active_flush'
                ant_transport_risk = 2.0  # ⭐ CRITICAL WINDOW
                recommendation = "🚨 CRITICAL: Apply ant barriers NOW! Ants transporting crawlers to new leaves."
                self.is_bud_break_active = True
                
            elif self.accumulated_warmth_days >= 5:
                phase = 'early_bud_break'
                ant_transport_risk = 1.8
                recommendation = "⚠️ HIGH ALERT: Buds swelling. Install sticky bands on trunks before flush."
                
        elif month >= 4:  # Post bud break
            phase = 'post_flush'
            ant_transport_risk = 1.2
            recommendation = "Monitor for crawler establishment on new leaves"
            
        return {
            'is_bud_break': self.is_bud_break_active,
            'phase': phase,
            'ant_transport_risk': ant_transport_risk,
            'recommendation': recommendation,
            'warmth_days_accumulated': self.accumulated_warmth_days
        }
