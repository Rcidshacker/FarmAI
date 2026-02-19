"""
Life Stage Tracker for Mealybug Development
Tracks egg → crawler → nymph → adult progression based on DD accumulation.
"""
import logging
from typing import Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class LifeStage:
    """Represents a single life stage"""
    name: str
    duration_dd: float  # Degree-days required
    vulnerability: float  # 0.0 to 1.0 (spray effectiveness)
    description: str

class LifeStageTracker:
    """
    Track mealybug life stages for M. hirsutus.
    Based on research: Total 250 DD generation time.
    """
    
    # M. hirsutus stage durations (from research)
    STAGES = {
        'egg': LifeStage('egg', 30, 0.3, 'Laid in waxy ovisac, 3-8 days'),
        'crawler': LifeStage('crawler', 35, 1.0, '1st instar, CRITICAL spray window'),
        'nymph_2': LifeStage('nymph_2', 65, 0.7, '2nd instar, feeding intensifies'),
        'nymph_3': LifeStage('nymph_3', 75, 0.8, '3rd instar, settlement'),
        'adult': LifeStage('adult', 45, 0.6, 'Reproductive stage')
    }
    
    TOTAL_GENERATION_DD = 250  # Sum of all stages
    
    def __init__(self):
        self.current_accumulated_dd = 0.0
        self.generation_count = 0
        
    def update(self, daily_dd: float) -> Dict:
        """
        Update accumulated DD and determine current dominant stage.
        
        Returns:
            {
                'current_stage': str,
                'stage_progress': float (0-1),
                'generation': int,
                'optimal_spray_window': bool,
                'vulnerability': float
            }
        """
        self.current_accumulated_dd += daily_dd
        
        # Check for generation completion
        if self.current_accumulated_dd >= self.TOTAL_GENERATION_DD:
            self.generation_count += 1
            self.current_accumulated_dd -= self.TOTAL_GENERATION_DD
            logger.info(f"Generation {self.generation_count} completed")
        
        # Determine current stage
        dd_progress = self.current_accumulated_dd
        for stage_key, stage in self.STAGES.items():
            if dd_progress < stage.duration_dd:
                stage_progress = dd_progress / stage.duration_dd
                
                return {
                    'current_stage': stage.name,
                    'stage_progress': stage_progress,
                    'generation': self.generation_count,
                    'optimal_spray_window': (stage_key == 'crawler'),  # ⭐ KEY!
                    'vulnerability': stage.vulnerability,
                    'description': stage.description
                }
            dd_progress -= stage.duration_dd
        
        # Fallback (adult stage)
        return {
            'current_stage': 'adult',
            'stage_progress': 1.0,
            'generation': self.generation_count,
            'optimal_spray_window': False,
            'vulnerability': 0.6,
            'description': 'Reproductive stage'
        }
    
    def reset(self):
        """Reset tracker for new season"""
        self.current_accumulated_dd = 0.0
        self.generation_count = 0
