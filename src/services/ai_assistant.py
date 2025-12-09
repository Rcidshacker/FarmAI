import json
from pathlib import Path
from typing import Dict, List
import logging
import os

logger = logging.getLogger(__name__)

class AIAssistantService:
    """
    AI Assistant for Farmers.
    Uses Knowledge Base to answer questions about pests, fertilizers, and schedules.
    """
    
    def __init__(self):
        # Try multiple paths for knowledge base (mobile and desktop)
        self.kb_paths = [
            Path("knowledge_base"),  # Desktop
            Path("../knowledge_base"),  # From src context
            Path("../../knowledge_base"),  # From nested context
        ]
        self.chemicals = {}
        self.pests = {}
        self._load_kb()
        
    def _load_kb(self):
        """Load knowledge base from multiple possible paths"""
        loaded = False
        
        for kb_path in self.kb_paths:
            try:
                chemical_file = kb_path / "chemical_compositions.json"
                pest_file = kb_path / "pest_database.json"
                
                if chemical_file.exists():
                    self.chemicals = json.loads(chemical_file.read_text())
                    logger.info(f"Loaded chemicals from {chemical_file}")
                    loaded = True
                    
                if pest_file.exists():
                    self.pests = json.loads(pest_file.read_text())
                    logger.info(f"Loaded pests from {pest_file}")
                    loaded = True
                    
                if loaded:
                    break
                    
            except Exception as e:
                logger.debug(f"Could not load KB from {kb_path}: {e}")
                continue
        
        # If files not found, use default inline knowledge base
        if not self.chemicals or not self.pests:
            logger.warning("Knowledge base files not found, using default inline knowledge")
            self._load_default_kb()
    
    def _load_default_kb(self):
        """Load default knowledge base inline for mobile/offline support"""
        self.chemicals = {
            "chemical_products": {
                "Imidacloprid": {
                    "name": "Imidacloprid",
                    "type": "Insecticide",
                    "target": ["Mealy Bug", "Scale Insects", "Aphids"],
                    "dosage": "3-5 ml per 10 liters of water",
                    "phi": 7,
                    "precautions": "Do not use during flowering. Use protective equipment."
                },
                "Thiamethoxam": {
                    "name": "Thiamethoxam",
                    "type": "Insecticide",
                    "target": ["Mealy Bug", "Whitefly", "Mites"],
                    "dosage": "2-3 ml per 10 liters of water",
                    "phi": 7,
                    "precautions": "Rotate with other insecticides. Avoid spray during hot hours."
                },
                "Spirotetramat": {
                    "name": "Spirotetramat",
                    "type": "Acaricide/Insecticide",
                    "target": ["Spider Mites", "Mealy Bug", "Scale Insects"],
                    "dosage": "2 ml per 10 liters of water",
                    "phi": 14,
                    "precautions": "Systemic action. Requires translocation. Avoid mixing with oils."
                },
                "Copper Hydroxide": {
                    "name": "Copper Hydroxide",
                    "type": "Fungicide",
                    "target": ["Fungal Diseases", "Bacterial Leaf Spots"],
                    "dosage": "20 grams per 10 liters of water",
                    "phi": 3,
                    "precautions": "Use at correct concentrations. Avoid overdose. Copper toxicity risk."
                },
                "Propiconazole": {
                    "name": "Propiconazole",
                    "type": "Fungicide",
                    "target": ["Powdery Mildew", "Leaf Spot", "Rust"],
                    "dosage": "1 ml per 10 liters of water",
                    "phi": 14,
                    "precautions": "Systemic fungicide. Do not mix with sulfur products."
                }
            }
        }
        
        self.pests = {
            "Mealy Bug": {
                "name": "Mealy Bug",
                "symptoms": "White waxy coating on leaves and stems, yellowing leaves, sticky residue",
                "control": "Use Imidacloprid or Spirotetramat. Maintain plant hygiene. Prune affected branches.",
                "season": "Year-round, peaks in summer"
            },
            "Spider Mites": {
                "name": "Spider Mites",
                "symptoms": "Fine webbing on leaves, yellowing, leaf drop, stippling pattern",
                "control": "Spray with neem oil or Spirotetramat. Increase humidity. Remove heavily infested leaves.",
                "season": "Hot and dry weather, summer"
            },
            "Powdery Mildew": {
                "name": "Powdery Mildew",
                "symptoms": "White powder coating on leaves, distorted growth, premature drop",
                "control": "Use sulfur dust or Propiconazole. Improve air circulation. Remove infected parts.",
                "season": "Cool and humid conditions"
            }
        }

    def get_response(self, query: str, context: Dict = None) -> Dict:
        """
        Generate a response to a farmer's query.
        """
        try:
            query = query.lower()
            response = {
                "text": "I'm not sure about that. Please consult an expert.",
                "related_topics": []
            }
            
            # 1. Pest Identification/Info
            for pest_name, data in self.pests.items():
                if pest_name.lower() in query:
                    symptoms = data.get('symptoms', 'Symptoms not available')
                    control = data.get('control', 'No control info')
                    response["text"] = f"**{pest_name}**: {symptoms}\n\n**Treatment**: {control}"
                    response["related_topics"] = ["Chemical Control", "Organic Control"]
                    return response

            # 2. Chemical Info - Handle both direct dict and chemical_products structure
            chemicals_to_search = self.chemicals
            if "chemical_products" in self.chemicals:
                chemicals_to_search = self.chemicals["chemical_products"]
            
            for chem_name, data in chemicals_to_search.items():
                if isinstance(data, dict) and chem_name.lower() in query:
                    targets = data.get('target', [])
                    if isinstance(targets, list):
                        targets_str = ', '.join(targets)
                    else:
                        targets_str = str(targets)
                    dosage = data.get('dosage', 'Check label')
                    response["text"] = f"**{chem_name}** is used for {targets_str}.\n\n**Dosage**: {dosage}"
                    if "precautions" in data:
                        response["text"] += f"\n\n**Precautions**: {data['precautions']}"
                    return response
                    
            # 3. General Intents
            if "weather" in query:
                response["text"] = "I can help with weather. Please check the 'Pest Risk' tab for the latest forecast and weather updates."
            elif "schedule" in query or "spray" in query:
                response["text"] = "Your spray schedule depends on the current pest risk. Check the 'Schedule' tab to see your optimized spray plan."
            elif "hello" in query or "hi" in query or "namaste" in query:
                response["text"] = "Namaste! I am your Farm Assistant. Ask me about pests, fertilizers, or current weather conditions."
                
            return response
        except Exception as e:
            logger.error(f"Error in get_response: {e}")
            return {
                "text": "I encountered an error processing your query. Please try again.",
                "related_topics": []
            }

