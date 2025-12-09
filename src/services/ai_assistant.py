import json
from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class AIAssistantService:
    """
    AI Assistant for Farmers.
    Uses Knowledge Base to answer questions about pests, fertilizers, and schedules.
    """
    
    def __init__(self):
        self.kb_path = Path("knowledge_base")
        self.chemicals = {}
        self.pests = {}
        self._load_kb()
        
    def _load_kb(self):
        try:
            if (self.kb_path / "chemical_compositions.json").exists():
                self.chemicals = json.loads((self.kb_path / "chemical_compositions.json").read_text())
            if (self.kb_path / "pest_database.json").exists():
                self.pests = json.loads((self.kb_path / "pest_database.json").read_text())
        except Exception as e:
            logger.error(f"Error loading KB for Assistant: {e}")

    def get_response(self, query: str, context: Dict = None) -> Dict:
        """
        Generate a response to a farmer's query.
        """
        query = query.lower()
        response = {
            "text": "I'm not sure about that. Please consult an expert.",
            "related_topics": []
        }
        
        # 1. Pest Identification/Info
        for pest_name, data in self.pests.items():
            if pest_name.lower() in query:
                response["text"] = f"**{pest_name}**: {data.get('symptoms', 'Symptoms not available')}. \n\n**Treatment**: {data.get('control', 'No control info')}."
                response["related_topics"] = ["Chemical Control", "Organic Control"]
                return response

        # 2. Chemical Info
        for chem_name, data in self.chemicals.items():
            if chem_name.lower() in query:
                response["text"] = f"**{chem_name}** is used for {', '.join(data.get('target', []))}. \n\n**Dosage**: {data.get('dosage', 'Check label')}."
                return response
                
        # 3. General Intents
        if "weather" in query:
            response["text"] = "I can help with weather. Please check the 'Weather' tab for the latest forecast."
        elif "schedule" in query or "spray" in query:
            response["text"] = "Your spray schedule depends on the current risk. Check the 'Plan' tab."
        elif "hello" in query or "hi" in query:
            response["text"] = "Namaste! I am your Custard Apple Farm Assistant. Ask me about pests, fertilizers, or weather."
            
        return response
