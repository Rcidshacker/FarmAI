"""
Automated Spray Scheduler using Reinforcement Learning
Learns optimal spray timing and application strategies based on:
- Weather forecasts
- Pest pressure history
- Treatment effectiveness feedback
- Economic costs
- Chemical resistance management
- Pre-harvest intervals (PHI)
- Proper spray gaps based on chemical properties
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json
import pickle
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ChemicalKnowledgeBase:
    """
    Manages chemical database for intelligent spray scheduling
    Includes resistance management, PHI, and spray intervals
    """
    
    def __init__(self, knowledge_path: str = "knowledge_base/chemical_compositions.json"):
        self.knowledge_path = Path(knowledge_path)
        self.chemicals = {}
        self.rotation_groups = {}
        self.load_knowledge_base()
    
    def load_knowledge_base(self):
        """Load chemical database from JSON with fallback"""
        try:
            # Try multiple paths
            paths_to_try = [
                self.knowledge_path,
                Path("../knowledge_base/chemical_compositions.json"),
                Path("../../knowledge_base/chemical_compositions.json"),
            ]
            
            for path in paths_to_try:
                if path.exists():
                    with open(path, 'r') as f:
                        data = json.load(f)
                        
                    # Parse chemical products
                    if 'chemical_products' in data:
                        self.chemicals = data['chemical_products']
                    
                    # Parse rotation groups
                    if 'chemical_categories' in data:
                        for category, groups in data['chemical_categories'].items():
                            for group_name, group_info in groups.items():
                                rotation_group = group_info.get('rotation_group', 'UN')
                                if rotation_group not in self.rotation_groups:
                                    self.rotation_groups[rotation_group] = []
                                self.rotation_groups[rotation_group].extend(
                                    group_info.get('active_ingredients', [])
                                )
                    
                    logger.info(f"Loaded {len(self.chemicals)} chemicals from {path}")
                    return
            
            # If no file found, use defaults
            logger.warning(f"Knowledge base file not found at {self.knowledge_path}, using defaults")
            self._load_default_chemicals()
            
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
            self._load_default_chemicals()
    
    def _load_default_chemicals(self):
        """Load default chemicals when file not found"""
        self.chemicals = {
            "imidacloprid_178sl": {
                "name": "Imidacloprid 17.8% SL",
                "dosage": "3-5 ml per 10L water",
                "phi": 7,
                "min_interval": 14,
                "target_pests": ["Mealy Bug", "Scale Insects", "Whitefly"]
            },
            "thiamethoxam_25wg": {
                "name": "Thiamethoxam 25% WG",
                "dosage": "2-3 grams per 10L water",
                "phi": 7,
                "min_interval": 14,
                "target_pests": ["Mealy Bug", "Whitefly", "Spider Mites"]
            },
            "neem_oil_1500ppm": {
                "name": "Neem Oil 1500 PPM",
                "dosage": "10-15 ml per 10L water",
                "phi": 0,
                "min_interval": 7,
                "target_pests": ["Scale Insects", "Mites", "Whitefly"]
            },
            "fish_oil_rosin_soap": {
                "name": "Fish Oil + Rosin Soap",
                "dosage": "20 ml per 10L water",
                "phi": 0,
                "min_interval": 5,
                "target_pests": ["Aphids", "Scale Insects", "Mites"]
            }
        }
        logger.info("Using default chemicals database")
    
    def get_chemical_by_pest_pressure(self, pest_pressure: float, 
                                     last_chemical: str = None,
                                     days_since_spray: int = 0) -> Dict:
        """
        Select appropriate chemical based on pest pressure and resistance management
        
        Args:
            pest_pressure: Current pest pressure (0-1)
            last_chemical: Last chemical used
            days_since_spray: Days since last application
        
        Returns:
            Dict with chemical details and recommendations
        """
        # Determine severity level
        if pest_pressure > 0.7:
            severity = "high"
        elif pest_pressure > 0.4:
            severity = "moderate"
        else:
            severity = "low"
        
        # Select chemical based on severity and rotation
        if severity == "high":
            # High pressure: Use systemic insecticides
            if last_chemical == "imidacloprid_178sl":
                # Rotate to different group
                chemical_id = "thiamethoxam_25wg"
            else:
                chemical_id = "imidacloprid_178sl"
        elif severity == "moderate":
            # Moderate: Use botanical/biological
            chemical_id = "neem_oil_1500ppm"
        else:
            # Low: Preventive measures
            chemical_id = "fish_oil_rosin_soap"
        
        # Get chemical details
        chemical = self.chemicals.get(chemical_id, {})
        
        # Check if minimum spray interval has passed
        min_interval = self._get_min_spray_interval(chemical_id)
        if days_since_spray < min_interval:
            return {
                'action': 'wait',
                'reason': f'Minimum spray interval not met ({days_since_spray}/{min_interval} days)',
                'days_to_wait': min_interval - days_since_spray
            }
        
        # Get PHI
        phi = self._extract_phi(chemical)
        
        return {
            'action': 'spray',
            'chemical_id': chemical_id,
            'chemical_name': chemical.get('trade_names', ['Unknown'])[0],
            'active_ingredient': chemical.get('active_ingredient', 'Unknown'),
            'dosage': chemical.get('dosage', {}).get('custard_apple', 'As per label'),
            'category': chemical.get('category', 'Unknown'),
            'target_pests': chemical.get('target_pests', []),
            'preharvest_interval': phi,
            'min_spray_interval': min_interval,
            'mixing_compatibility': chemical.get('mixing_compatibility', 'Unknown'),
            'precautions': chemical.get('precautions', [])
        }
    
    def _get_min_spray_interval(self, chemical_id: str) -> int:
        """Get minimum days between applications"""
        intervals = {
            'imidacloprid_178sl': 21,  # 3 weeks
            'thiamethoxam_25wg': 21,
            'mancozeb_75wp': 10,  # Can be more frequent
            'carbendazim_50wp': 14,
            'neem_oil_1500ppm': 7,  # Botanical, can be frequent
            'copper_oxychloride_50wp': 10,
            'fish_oil_rosin_soap': 5  # Preventive, frequent
        }
        return intervals.get(chemical_id, 14)  # Default 2 weeks
    
    def _extract_phi(self, chemical: Dict) -> int:
        """Extract pre-harvest interval in days"""
        phi_str = chemical.get('preharvest_interval', '15 days')
        try:
            return int(''.join(filter(str.isdigit, phi_str)))
        except:
            return 15  # Default
    
    def get_rotation_plan(self, num_sprays: int = 4) -> List[str]:
        """
        Get chemical rotation plan to prevent resistance
        
        Args:
            num_sprays: Number of sprays in season
        
        Returns:
            List of chemical IDs to use in sequence
        """
        # Rotation strategy: alternate between groups
        rotation = []
        groups = ['4A', 'M3', '1B', 'UN']  # Neonicotinoid, Dithiocarbamate, Organophosphate, Botanical
        chemicals_by_group = {
            '4A': ['imidacloprid_178sl', 'thiamethoxam_25wg'],
            'M3': ['mancozeb_75wp'],
            '1B': ['chlorpyrifos_20ec'],
            'UN': ['neem_oil_1500ppm', 'fish_oil_rosin_soap']
        }
        
        for i in range(num_sprays):
            group = groups[i % len(groups)]
            chemicals = chemicals_by_group.get(group, [])
            if chemicals:
                rotation.append(chemicals[i % len(chemicals)])
        
        return rotation


class SpraySchedulerEnvironment:
    """
    RL Environment for spray scheduling with proper chemical knowledge integration
    
    State: [pest_pressure, weather_conditions, days_since_last_spray, growth_stage]
    Action: [spray_now, wait_3_days, wait_7_days, wait_14_days]
    Reward: Based on pest control effectiveness, cost, environmental impact, and resistance management
    """
    
    def __init__(self, weather_forecast: pd.DataFrame, initial_pest_pressure: float = 0.3,
                 knowledge_base: ChemicalKnowledgeBase = None):
        self.weather_forecast = weather_forecast
        self.current_day = 0
        self.pest_pressure = initial_pest_pressure
        self.days_since_spray = 21  # Start with proper interval
        self.growth_stage = 0  # 0: Vegetative, 1: Flowering, 2: Fruit Dev, 3: Maturity
        self.total_sprays = 0
        self.total_cost = 0
        self.disease_severity = 0.2
        self.last_chemical = None
        self.spray_history = []
        
        # Initialize knowledge base
        self.kb = knowledge_base or ChemicalKnowledgeBase()
        
        # Economic parameters (in INR)
        self.spray_cost = 500  # Per application
        self.yield_loss_per_pressure = 1000  # Loss per 0.1 pest pressure
        
    def reset(self):
        """Reset environment to initial state"""
        self.current_day = 0
        self.pest_pressure = 0.3
        self.days_since_spray = 21
        self.total_sprays = 0
        self.total_cost = 0
        self.disease_severity = 0.2
        self.last_chemical = None
        self.spray_history = []
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """Get current state vector"""
        if self.current_day >= len(self.weather_forecast):
            self.current_day = len(self.weather_forecast) - 1
        
        weather = self.weather_forecast.iloc[self.current_day]
        
        # FIX: Handle missing 'temp' column safely by using average of max/min if needed
        if 'temp' not in weather and 'tempmax' in weather:
            current_temp = (weather['tempmax'] + weather['tempmin']) / 2
        else:
            current_temp = weather.get('temp', 25.0)
        
        state = np.array([
            self.pest_pressure,
            self.disease_severity,
            current_temp / 40.0,  # Normalize
            weather['humidity'] / 100.0,
            weather.get('precip', 0) / 50.0,  # Normalize
            self.days_since_spray / 30.0,  # Normalize
            self.growth_stage / 3.0,
            self.total_sprays / 10.0  # Normalize
        ], dtype=np.float32)
        
        return state
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take action and return next state, reward, done, info
        
        Actions:
        0: Spray now (with proper chemical selection)
        1: Wait 3 days
        2: Wait 7 days
        3: Wait 14 days
        """
        weather = self.weather_forecast.iloc[self.current_day]
        
        # FIX: Safe temp access
        if 'temp' not in weather and 'tempmax' in weather:
            temp = (weather['tempmax'] + weather['tempmin']) / 2
        else:
            temp = weather.get('temp', 25.0)
        
        humidity = weather['humidity']
        rain = weather.get('precip', 0)
        
        # Use pre-calculated risk score if available (from Biological Model)
        if 'mealybug_risk_score' in weather:
            pressure_increase = weather['mealybug_risk_score'] * 0.2 # Scale it down for daily increment
        else:
            # Pest growth model (simplified fallback)
            if 25 <= temp <= 35 and humidity > 60:
                pressure_increase = 0.1  # Favorable conditions
            elif temp > 35 or temp < 20:
                pressure_increase = 0.03  # Unfavorable
            else:
                pressure_increase = 0.05  # Moderate
        
        # Rain reduces spray effectiveness
        if rain > 10:
            pressure_increase += 0.05
        
        reward = 0
        info = {}
        
        if action == 0:  # Spray now
            # Get chemical recommendation from knowledge base
            chem_recommendation = self.kb.get_chemical_by_pest_pressure(
                self.pest_pressure, 
                self.last_chemical,
                self.days_since_spray
            )
            
            # Check if spray is allowed (minimum interval check)
            if chem_recommendation.get('action') == 'wait':
                info['action_taken'] = 'Spray Rejected'
                info['reason'] = chem_recommendation['reason']
                reward -= 50  # Penalty for trying to spray too early
                
            else:
                # Check if weather is suitable for spraying
                if rain > 5:
                    effectiveness = 0.4
                    info['spray_quality'] = 'Poor (Rain)'
                    reward -= 30  # Penalty for spraying in rain
                elif humidity > 85:
                    effectiveness = 0.6
                    info['spray_quality'] = 'Moderate (High Humidity)'
                elif temp > 35:
                    effectiveness = 0.5
                    info['spray_quality'] = 'Poor (High Temperature)'
                    reward -= 20
                else:
                    effectiveness = 0.85
                    info['spray_quality'] = 'Good'
                
                # Get chemical efficacy
                chemical_efficacy = self._get_chemical_efficacy(chem_recommendation)
                total_effectiveness = effectiveness * chemical_efficacy
                
                # Apply spray effect
                self.pest_pressure = max(0, self.pest_pressure * (1 - total_effectiveness))
                self.disease_severity = max(0, self.disease_severity * (1 - total_effectiveness * 0.8))
                
                # Update cost tracking
                cost = chem_recommendation.get('cost_per_liter', 500) * 5  # Assume 5L per spray
                self.total_cost += cost
                self.total_sprays += 1
                self.days_since_spray = 0
                self.last_chemical = chem_recommendation.get('chemical_id')
                
                # Reward for effective spray
                reward += 100 * total_effectiveness
                
                # Bonus for spraying at optimal time (0.5-0.7 pest pressure)
                if 0.5 <= self.pest_pressure <= 0.7:
                    reward += 60  # Optimal timing
                elif self.pest_pressure > 0.7:
                    reward += 40  # High pressure, needed but late
                
                # Bonus for proper chemical rotation
                if self.total_sprays > 1:
                    reward += 20  # Good resistance management
                
                info['action_taken'] = 'Sprayed'
                info['chemical'] = chem_recommendation
        
        else:  # Wait
            wait_days = [3, 7, 14][action - 1]  # Changed from [1, 3, 7] to [3, 7, 14]
            
            # Pest pressure increases while waiting
            for _ in range(wait_days):
                self.pest_pressure = min(1.0, self.pest_pressure + pressure_increase)
                self.disease_severity = min(1.0, self.disease_severity + pressure_increase * 0.5)
                self.days_since_spray += 1
                self.current_day += 1
                
                if self.current_day >= len(self.weather_forecast):
                    break
            
            # Reward for waiting (saving costs)
            reward += 40 * wait_days
            
            # Smart penalty based on pest pressure
            if self.pest_pressure > 0.8:
                reward -= 80  # Critical pest pressure - should have sprayed
            elif self.pest_pressure > 0.6:
                reward -= 30  # High pressure - action needed soon
            elif self.pest_pressure < 0.3:
                reward += 30  # Good decision to wait - no need to spray
            
            info['action_taken'] = f'Waited {wait_days} days'
        
        # Advance day
        if action == 0:
            self.current_day += 1
            self.days_since_spray += 1
        
        # Update growth stage (simplified)
        if self.current_day > 90:
            self.growth_stage = 3  # Maturity
        elif self.current_day > 60:
            self.growth_stage = 2  # Fruit Development
        elif self.current_day > 30:
            self.growth_stage = 1  # Flowering
        else:
            self.growth_stage = 0  # Vegetative
        
        # Episode ends after forecast period or if pest pressure too high
        done = (self.current_day >= len(self.weather_forecast) - 1) or (self.pest_pressure > 0.9)
        
        # Calculate yield loss
        yield_loss = self.pest_pressure * self.yield_loss_per_pressure * len(self.weather_forecast)
        info['yield_loss'] = yield_loss
        info['total_cost'] = self.total_cost
        info['total_sprays'] = self.total_sprays
        info['pest_pressure'] = self.pest_pressure
        
        next_state = self.get_state()
        
        return next_state, reward, done, info
    
    def _get_chemical_efficacy(self, chem_recommendation: Dict) -> float:
        """Get chemical efficacy multiplier based on type"""
        category = chem_recommendation.get('category', 'Unknown')
        
        efficacy_map = {
            'Insecticide': 0.95,
            'Fungicide': 0.90,
            'Botanical Insecticide': 0.75,
            'Biocontrol': 0.65
        }
        
        return efficacy_map.get(category, 0.80)
        info['total_sprays'] = self.total_sprays
        info['pest_pressure'] = self.pest_pressure
        
        next_state = self.get_state()
        
        return next_state, reward, done, info


class QLearningSprayScheduler:
    """
    Q-Learning agent for spray scheduling
    Uses tabular Q-learning with state discretization
    """
    
    def __init__(self, state_bins: List[int] = [10, 10, 5, 5, 5, 10, 4, 5], 
                 n_actions: int = 4, learning_rate: float = 0.1, 
                 discount_factor: float = 0.95, epsilon: float = 0.2):
        
        self.state_bins = state_bins
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        
        # Initialize Q-table (discretized state space)
        self.q_table = {}
        
    def discretize_state(self, state: np.ndarray) -> Tuple:
        """Discretize continuous state to bins"""
        discrete_state = []
        for i, (value, bins) in enumerate(zip(state, self.state_bins)):
            bin_idx = int(np.clip(value * bins, 0, bins - 1))
            discrete_state.append(bin_idx)
        return tuple(discrete_state)
    
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        discrete_state = self.discretize_state(state)
        
        # Epsilon-greedy
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        # Initialize Q-values if state not seen
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.n_actions)
        
        # Choose best action
        return int(np.argmax(self.q_table[discrete_state]))
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool):
        """Update Q-values using Q-learning update rule"""
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        
        # Initialize if not seen
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.n_actions)
        if discrete_next_state not in self.q_table:
            self.q_table[discrete_next_state] = np.zeros(self.n_actions)
        
        # Q-learning update
        current_q = self.q_table[discrete_state][action]
        
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[discrete_next_state])
            target_q = reward + self.gamma * max_next_q
        
        # Update Q-value
        self.q_table[discrete_state][action] += self.lr * (target_q - current_q)
    
    def train(self, env: SpraySchedulerEnvironment, episodes: int = 1000) -> List[float]:
        """Train the agent"""
        episode_rewards = []
        
        logger.info(f"Starting training for {episodes} episodes...")
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.get_action(state, training=True)
                next_state, reward, done, info = env.step(action)
                
                self.update(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
            
            episode_rewards.append(total_reward)
            
            # Decay epsilon
            self.epsilon = max(0.01, self.epsilon * 0.995)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                logger.info(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.2f}, "
                          f"Epsilon: {self.epsilon:.3f}")
        
        # Save training results
        self._save_training_results(episode_rewards, episodes)
        
        return episode_rewards
    
    def _save_training_results(self, episode_rewards: List[float], episodes: int):
        """
        Save training results including metrics and plots
        
        Args:
            episode_rewards: List of rewards per episode
            episodes: Total number of episodes
        """
        import matplotlib.pyplot as plt
        from pathlib import Path
        from datetime import datetime
        
        # Create results directory
        results_dir = Path("training_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate metrics
        avg_reward_last_100 = np.mean(episode_rewards[-100:])
        avg_reward_first_100 = np.mean(episode_rewards[:100])
        max_reward = max(episode_rewards)
        min_reward = min(episode_rewards)
        final_reward = episode_rewards[-1]
        
        # Save metrics to JSON
        metrics_file = results_dir / f"spray_scheduler_metrics_{timestamp}.json"
        metrics = {
            'episodes': episodes,
            'avg_reward_last_100': float(avg_reward_last_100),
            'avg_reward_first_100': float(avg_reward_first_100),
            'improvement': float(avg_reward_last_100 - avg_reward_first_100),
            'max_reward': float(max_reward),
            'min_reward': float(min_reward),
            'final_reward': float(final_reward),
            'final_epsilon': float(self.epsilon),
            'learning_rate': float(self.lr),
            'discount_factor': self.gamma,
            'timestamp': timestamp
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Training metrics saved to {metrics_file}")
        
        # Plot training curve
        try:
            plt.figure(figsize=(12, 6))
            
            # Raw rewards
            plt.subplot(1, 2, 1)
            plt.plot(episode_rewards, alpha=0.3, label='Episode Reward')
            
            # Moving average
            window = 100
            moving_avg = [np.mean(episode_rewards[max(0, i-window):i+1]) 
                         for i in range(len(episode_rewards))]
            plt.plot(moving_avg, label=f'{window}-Episode Moving Avg', linewidth=2)
            
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title('Spray Scheduler Training Progress')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Distribution
            plt.subplot(1, 2, 2)
            plt.hist(episode_rewards, bins=50, alpha=0.7, edgecolor='black')
            plt.axvline(avg_reward_last_100, color='r', linestyle='--', 
                       label=f'Avg Last 100: {avg_reward_last_100:.2f}')
            plt.xlabel('Episode Reward')
            plt.ylabel('Frequency')
            plt.title('Reward Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            plot_file = results_dir / f"spray_scheduler_training_{timestamp}.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Training plots saved to {plot_file}")
            
        except Exception as e:
            logger.warning(f"Could not create training plots: {e}")
        
        # Print summary
        logger.info("=" * 60)
        logger.info("TRAINING SUMMARY - Spray Scheduler (Q-Learning)")
        logger.info("=" * 60)
        logger.info(f"Episodes: {episodes}")
        logger.info(f"Initial Avg Reward (first 100): {avg_reward_first_100:.2f}")
        logger.info(f"Final Avg Reward (last 100): {avg_reward_last_100:.2f}")
        logger.info(f"Improvement: {avg_reward_last_100 - avg_reward_first_100:.2f}")
        logger.info(f"Max Reward: {max_reward:.2f}")
        logger.info(f"Final Epsilon: {self.epsilon:.4f}")
        logger.info("=" * 60)
    
    def generate_schedule(self, weather_forecast: pd.DataFrame, 
                         initial_pest_pressure: float = 0.3) -> List[Dict]:
        """
        Generate spray schedule for given weather forecast with proper chemical rotation
        
        Returns:
            List of spray events with dates, chemicals, and reasoning
        """
        env = SpraySchedulerEnvironment(weather_forecast, initial_pest_pressure)
        state = env.reset()
        
        schedule = []
        done = False
        current_date = datetime.now()
        
        while not done:
            action = self.get_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            
            if info['action_taken'] == 'Sprayed' and 'chemical' in info:
                chem = info['chemical']
                
                schedule.append({
                    'date': current_date + timedelta(days=env.current_day),
                    'action': 'Spray Application',
                    'chemical_name': chem.get('chemical_name', 'Unknown'),
                    'active_ingredient': chem.get('active_ingredient', 'Unknown'),
                    'dosage': chem.get('dosage', 'As per label'),
                    'target_pests': chem.get('target_pests', []),
                    'preharvest_interval': chem.get('preharvest_interval', 15),
                    'min_next_spray': chem.get('min_spray_interval', 14),
                    'pest_pressure': round(info['pest_pressure'], 2),
                    'spray_quality': info['spray_quality'],
                    'weather': {
                        'temp': round(weather_forecast.iloc[env.current_day].get('temp', 
                                (weather_forecast.iloc[env.current_day].get('tempmax', 0) + 
                                 weather_forecast.iloc[env.current_day].get('tempmin', 0)) / 2), 1),
                        'humidity': round(weather_forecast.iloc[env.current_day]['humidity'], 0),
                        'rainfall': round(weather_forecast.iloc[env.current_day].get('precip', 0), 1)
                    },
                    'precautions': chem.get('precautions', []),
                    'mixing_compatibility': chem.get('mixing_compatibility', 'Unknown'),
                    'reasoning': f"Pest pressure: {info['pest_pressure']:.2f}, Quality: {info['spray_quality']}, "
                                f"Chemical rotation: {chem.get('category', 'Unknown')}"
                })
            
            state = next_state
        
        # Add summary with recommendations
        schedule.append({
            'summary': {
                'total_sprays': info['total_sprays'],
                'total_cost': info['total_cost'],
                'estimated_yield_loss': info['yield_loss'],
                'final_pest_pressure': round(info['pest_pressure'], 2),
                'spray_interval': f"{round(90 / max(1, info['total_sprays']))} days average",
                'resistance_management': 'Chemical rotation applied' if info['total_sprays'] > 1 else 'Single application',
                'chemicals_used': list(set([s['chemical_name'] for s in schedule if 'chemical_name' in s]))
            }
        })
        
        return schedule
    
    def save_model(self, filepath: str = "models/spray_scheduler.pkl"):
        """Save trained Q-table"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'state_bins': self.state_bins,
                'n_actions': self.n_actions,
                'epsilon': self.epsilon
            }, f)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str = "models/spray_scheduler.pkl"):
        """Load trained Q-table"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.state_bins = data['state_bins']
            self.n_actions = data['n_actions']
            self.epsilon = data['epsilon']
        logger.info(f"Model loaded from {filepath}")


class AutomatedSprayManager:
    """
    High-level manager for automated spray scheduling
    Integrates with weather API and sends notifications/alerts
    """
    
    def __init__(self, scheduler: QLearningSprayScheduler = None):
        self.scheduler = scheduler or QLearningSprayScheduler()
        self.current_schedule = []
        
    def create_schedule(self, location: str, days_ahead: int = 30) -> Dict:
        """
        Create optimized spray schedule for a location
        
        Args:
            location: Farm location
            days_ahead: Days to plan ahead
            
        Returns:
            Dict with schedule and recommendations
        """
        try:
            # Fetch weather forecast (integrate with IMD API)
            from src.data_sources.external_data_fetcher import IMDWeatherFetcher
            
            weather_fetcher = IMDWeatherFetcher()
            forecast_df = weather_fetcher.get_forecast(location, days=days_ahead)
            
            # Ensure we have valid forecast data
            if forecast_df is None or (isinstance(forecast_df, object) and hasattr(forecast_df, 'empty') and forecast_df.empty):
                logger.warning("Could not fetch forecast, using default")
                # Generate dummy forecast
                forecast_df = self._generate_dummy_forecast(days_ahead)
                
            # ---------------------------------------------------------
            # FIX APPLIED HERE: Ensure 'temp' column exists
            # The RL Environment requires a specific 'temp' column.
            # ---------------------------------------------------------
            if 'temp' not in forecast_df.columns:
                if 'tempmax' in forecast_df.columns and 'tempmin' in forecast_df.columns:
                    # Calculate average temp from max/min
                    forecast_df['temp'] = (forecast_df['tempmax'] + forecast_df['tempmin']) / 2
                    logger.info("Calculated 'temp' column from tempmax/tempmin")
                else:
                    # Fallback default if completely missing
                    logger.warning("'temp' column missing in forecast. Using default 28.0")
                    forecast_df['temp'] = 28.0
            # ---------------------------------------------------------
                
            # Integrate Biological Risk Model
            try:
                from src.models.biological_risk_model import BiologicalRiskModel
                bio_model = BiologicalRiskModel()
                # Ensure columns match what BiologicalRiskModel expects
                # It expects: datetime, tempmax, tempmin, precip, humidity
                # forecast_df usually has: datetime, temp, humidity, precip, wind_speed
                
                # Map columns if needed
                if 'tempmax' not in forecast_df.columns:
                    forecast_df['tempmax'] = forecast_df['temp'] + 5
                    forecast_df['tempmin'] = forecast_df['temp'] - 5
                
                forecast_df = bio_model.calculate_risk_series(forecast_df)
                logger.info("Biological risk scores integrated into forecast")
            except Exception as e:
                logger.error(f"Failed to integrate biological risk model: {e}")
            
            # Generate schedule
            schedule_with_summary = self.scheduler.generate_schedule(forecast_df)
            
            # Separate schedule items from summary
            if schedule_with_summary and len(schedule_with_summary) > 0:
                # Last item contains summary
                if isinstance(schedule_with_summary[-1], dict) and 'summary' in schedule_with_summary[-1]:
                    schedule = schedule_with_summary[:-1]
                    summary = schedule_with_summary[-1]['summary']
                else:
                    schedule = schedule_with_summary
                    summary = {
                        'total_sprays': 0,
                        'total_cost': 0,
                        'estimated_yield_loss': 0,
                        'final_pest_pressure': 0,
                        'spray_interval': 'N/A',
                        'resistance_management': 'N/A',
                        'chemicals_used': []
                    }
            else:
                schedule = []
                summary = {
                    'total_sprays': 0,
                    'total_cost': 0,
                    'estimated_yield_loss': 0,
                    'final_pest_pressure': 0,
                    'spray_interval': 'N/A',
                    'resistance_management': 'N/A',
                    'chemicals_used': []
                }
            
            return {
                'location': location,
                'forecast_period': f"{days_ahead} days",
                'schedule': schedule,
                'summary': summary,
                'created_at': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in create_schedule: {e}", exc_info=True)
            # Return empty schedule instead of crashing
            return {
                'location': location,
                'forecast_period': f"{days_ahead} days",
                'schedule': [],
                'summary': {
                    'total_sprays': 0,
                    'total_cost': 0,
                    'estimated_yield_loss': 0,
                    'final_pest_pressure': 0,
                    'spray_interval': 'N/A',
                    'resistance_management': 'N/A',
                    'chemicals_used': []
                },
                'created_at': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def _generate_dummy_forecast(self, days: int) -> pd.DataFrame:
        """Generate dummy weather forecast for testing"""
        dates = pd.date_range(start=datetime.now(), periods=days, freq='D')
        
        # Simulate realistic Maharashtra weather
        forecast = []
        for i, date in enumerate(dates):
            forecast.append({
                'datetime': date,
                'temp': np.random.normal(28, 4),
                'humidity': np.random.normal(70, 12),
                'precip': max(0, np.random.exponential(3)),
                'wind_speed': np.random.normal(5, 2)
            })
        
        return pd.DataFrame(forecast)
    
    def get_next_spray_date(self) -> Dict:
        """Get next scheduled spray date and details"""
        if not self.current_schedule:
            return {'message': 'No schedule created yet'}
        
        today = datetime.now()
        for event in self.current_schedule:
            if event.get('date') and event['date'] > today:
                return {
                    'next_spray_date': event['date'].strftime('%Y-%m-%d'),
                    'days_until': (event['date'] - today).days,
                    'expected_conditions': event['weather'],
                    'reasoning': event['reasoning']
                }
        
        return {'message': 'No upcoming sprays in schedule'}
    
    def send_alerts(self, schedule: List[Dict]) -> List[str]:
        """
        Generate alerts for upcoming spray events
        
        Returns:
            List of alert messages
        """
        alerts = []
        today = datetime.now()
        
        for event in schedule:
            if not event.get('date'):
                continue
            
            days_until = (event['date'] - today).days
            
            if days_until == 1:
                alerts.append(
                    f"🔔 SPRAY ALERT: Spray scheduled for tomorrow "
                    f"({event['date'].strftime('%d %b')}). "
                    f"Weather: {event['weather']['temp']:.1f}°C, "
                    f"{event['weather']['humidity']:.0f}% humidity. "
                    f"Quality expected: {event['spray_quality']}"
                )
            elif days_until == 3:
                alerts.append(
                    f"⚠️ ADVANCE NOTICE: Spray recommended in 3 days "
                    f"({event['date'].strftime('%d %b')}). "
                    f"Prepare equipment and materials."
                )
        
        return alerts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("AUTOMATED SPRAY SCHEDULER - RL TRAINING")
    print("=" * 70)
    
    # Create dummy weather forecast for training
    np.random.seed(42)
    days = 90
    dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
    
    weather_df = pd.DataFrame({
        'datetime': dates,
        'temp': np.random.normal(28, 5, days),
        'humidity': np.random.normal(70, 15, days),
        'precip': np.maximum(0, np.random.exponential(5, days)),
        'wind_speed': np.random.normal(5, 2, days)
    })
    
    print(f"\n✓ Created {days}-day weather forecast for training")
    
    # Create environment and agent
    env = SpraySchedulerEnvironment(weather_df)
    agent = QLearningSprayScheduler()
    
    # Train agent
    print(f"\n🤖 Training RL agent...")
    rewards = agent.train(env, episodes=500)
    
    print(f"\n✓ Training completed!")
    print(f"   Final average reward: {np.mean(rewards[-100:]):.2f}")
    
    # Save model
    agent.save_model()
    
    # Test schedule generation
    print("\n" + "=" * 70)
    print("GENERATING OPTIMAL SPRAY SCHEDULE")
    print("=" * 70)
    
    manager = AutomatedSprayManager(agent)
    schedule_result = manager.create_schedule("Pune", days_ahead=30)
    
    print(f"\n📅 Spray Schedule for {schedule_result['location']}")
    print(f"   Forecast Period: {schedule_result['forecast_period']}")
    print(f"\n   Recommended Spray Dates:")
    
    for i, event in enumerate(schedule_result['schedule'], 1):
        print(f"\n   {i}. {event['date'].strftime('%d %b %Y')}")
        print(f"      Weather: {event['weather']['temp']:.1f}°C, "
              f"{event['weather']['humidity']:.0f}% humidity, "
              f"{event['weather']['rainfall']:.1f}mm rain")
        print(f"      Quality: {event['spray_quality']}")
        print(f"      Reasoning: {event['reasoning']}")
    
    summary = schedule_result['summary']
    print(f"\n{'─' * 70}")
    print(f"SUMMARY:")
    print(f"{'─' * 70}")
    print(f"   Total Sprays: {summary['total_sprays']}")
    print(f"   Total Cost: ₹{summary['total_cost']:.0f}")
    print(f"   Estimated Yield Protection: ₹{10000 - summary['estimated_yield_loss']:.0f}")
    print(f"   Final Pest Control: {(1 - summary['final_pest_pressure']) * 100:.1f}%")
    
    # Generate alerts
    print(f"\n{'─' * 70}")
    print(f"ALERTS:")
    print(f"{'─' * 70}")
    alerts = manager.send_alerts(schedule_result['schedule'])
    for alert in alerts[:3]:  # Show first 3 alerts
        print(f"   {alert}")
    
    print("\n" + "=" * 70)
    print("AUTOMATED SPRAY SCHEDULER READY!")
    print("=" * 70)
