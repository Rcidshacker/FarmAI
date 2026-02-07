import json
from pathlib import Path
from typing import Dict, List, Optional
import logging
import os
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class AIAssistantService:
    """
    Hybrid AI Assistant for Farmers.
    - Online: Uses Llama 3.3 (via OpenRouter) for intelligent, context-aware answers.
    - Offline: Uses local Knowledge Base for basic keyword matching.
    """
    
    def __init__(self):
        self.kb_path = Path("knowledge_base")
        self.chemicals = {}
        self.pests = {}
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model = "meta-llama/llama-3.3-70b-instruct:free"
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
        Generate response using LLM if available, else Fallback.
        """
        # Check for Offline Mode or Missing Key
        is_offline = os.getenv("OFFLINE_MODE", "False").lower() == "true"
        
        if not is_offline and self.api_key:
            try:
                return self._get_llm_response(query, context)
            except Exception as e:
                logger.error(f"LLM Error: {e}. Switching to Offline Fallback.")
                
        return self._get_offline_response(query)

    def _get_llm_response(self, query: str, context: Dict) -> Dict:
        """Call OpenRouter LLM"""
        
        # 1. Build System Prompt with Context
        ctx_str = ""
        if context:
            loc = context.get('location', 'Unknown')
            weather = context.get('weather', {})
            soil = context.get('soil', {})
            stage = context.get('crop_stage', 'Unknown')
            
            ctx_str = (
                f"Current Context:\n"
                f"- Location: {loc}\n"
                f"- Crop Stage: {stage}\n"
                f"- Weather: {weather.get('temp', 'N/A')}°C, {weather.get('humidity', 'N/A')}% Humidity, {weather.get('rainfall', 'N/A')}mm Rain\n"
                f"- Soil: {soil.get('type', 'N/A')} ({soil.get('moisture', 'N/A')}% Moisture)\n"
            )

        system_prompt = (
            "You are 'FarmAI', an expert agricultural consultant for Custard Apple (Sitaphal) farmers in Maharashtra, India. "
            "Your goal is to provide practical, scientific, and cost-effective advice.\n"
            "GUIDELINES:\n"
            "1. Be concise but helpful. Avoid fluff.\n"
            "2. Use the provided Context (Weather, Soil, Stage) to tailor your advice. "
            "   (e.g., if it's raining, advise against spraying).\n"
            "3. If suggesting chemicals, always mention dosage and safety.\n"
            "4. Be polite and encouraging."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{ctx_str}\n\nUser Question: {query}"}
        ]

        # 2. API Call
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000", # Required by OpenRouter
            "X-Title": "FarmAI"
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 500
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            return {
                "text": content,
                "source": "Llama 3.3 (Online)",
                "related_topics": ["Pest Control", "Weather Management"] # Simplified for now
            }
        else:
            raise Exception(f"API Failed: {response.status_code} - {response.text}")

    def _get_offline_response(self, query: str) -> Dict:
        """Legacy Keyword Matching (Offline Fallback)"""
        query = query.lower()
        response = {
            "text": "I'm currently offline and couldn't find a specific answer in my local database. Please try connecting to the internet for AI assistance.",
            "source": "Offline Database",
            "related_topics": []
        }
        
        # 1. Pest Identification/Info
        for pest_name, data in self.pests.items():
            if pest_name.lower() in query:
                response["text"] = f"**{pest_name}** (Offline Info): {data.get('symptoms', 'Symptoms not available')}. \n\n**Treatment**: {data.get('control', 'No control info')}."
                response["related_topics"] = ["Chemical Control", "Organic Control"]
                return response

        # 2. Chemical Info
        for chem_name, data in self.chemicals.items():
            if chem_name.lower() in query:
                response["text"] = f"**{chem_name}** (Offline Info) is used for {', '.join(data.get('target', []))}. \n\n**Dosage**: {data.get('dosage', 'Check label')}."
                return response
                
        # 3. General Intents
        if "weather" in query:
            response["text"] = "I cannot fetch live analysis offline, but you can check the 'Weather' tab for the cached forecast."
        elif "schedule" in query or "spray" in query:
            response["text"] = "Your spray schedule is available in the 'Plan' tab."
        elif "hello" in query or "hi" in query:
            response["text"] = "Namaste! I am your Offline Farm Assistant. I can answer basic questions about pests and chemicals needed."
            
        return response
