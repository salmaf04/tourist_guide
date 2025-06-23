"""
LLM Service for language model operations
Handles LLM API calls and response parsing
"""

import re
import logging
from typing import Dict, List, Optional
from agent_generator.mistral_client import MistralClient

logger = logging.getLogger(__name__)

class LLMService:
    """Handles LLM operations and response processing"""
    
    def __init__(self):
        """Initialize LLM service with Mistral client"""
        self.mistral_client = MistralClient()
    
    def call_mistral_llm(self, prompt: str, system: Optional[str] = None) -> str:
        """Call Mistral API for LLM response using the Mistral client."""
        try:
            logger.info("🤖 Calling Mistral API...")
            response = self.mistral_client.generate(prompt, system)
            logger.info(f"🤖 Mistral API response received: {len(response) if response else 0} characters")
            
            if response and response != "[Error de generación]" and response != "[Límite de solicitudes excedido]":
                logger.info("✅ Mistral API call successful")
                return response
            else:
                logger.error(f"❌ Mistral client failed to generate response: {response}")
                raise RuntimeError(f"Mistral client failed to generate response: {response}")
        except Exception as e:
            logger.error(f"❌ Error calling Mistral client: {e}")
            raise RuntimeError(f"Failed to call Mistral LLM: {e}")

    def get_llm_recommendation_prompt(self, places: List[Dict], user_preferences: Dict) -> str:
        """Generate a prompt for the LLM to estimate time only (categories come from database)."""
        if not places:
            return "No places found matching the criteria."

        prompt = f"""
Based on the following tourist preferences and available places, estimate how much time (in hours) the tourist would be interested in spending at each location.

TOURIST PROFILE:
- Visiting: {user_preferences.get('city', 'Unknown city')}
- Available hours for tourism: {user_preferences.get('available_hours', 'Unknown')} hours
- Maximum travel distance: {user_preferences.get('max_distance', 'Unknown')} km
- Transportation modes: {', '.join(user_preferences.get('transport_modes', []))}

CATEGORY INTERESTS (1-5 scale):
"""
        category_interest = user_preferences.get('category_interest', {})
        for category, score in category_interest.items():
            prompt += f"- {category.capitalize()}: {score}/5\n"

        if user_preferences.get('user_notes'):
            prompt += f"\nADDITIONAL NOTES: {user_preferences['user_notes']}\n"

        prompt += "\nAVAILABLE PLACES:\n"
        for i, place in enumerate(places):
            prompt += f"""
{i+1}. {place['name']}
   - Type: {place.get('type', 'Unknown')}
   - Description: {place.get('description', 'No description')}
   - Category: {place.get('category', 'Unknown')}
   - Estimated Visit Duration: {place.get('estimatedVisitDuration', 'Unknown')}
"""

        prompt += """
TASK: For each place listed above, provide only the estimated time (in hours) the tourist would be interested in spending there based on their preferences.

Consider:
- The tourist's category interests and ratings
- The type and appeal of each place
- The tourist's additional notes and preferences
- The suggested visit duration for each place

Provide your response in the following format:
Place 1: X.X hours - Brief reasoning
Place 2: X.X hours - Brief reasoning
...

Be realistic with time estimates based on the tourist's interests and the nature of each place.
"""
        return prompt

    def parse_llm_time_estimates(self, llm_response: str, places: List[Dict]) -> List[float]:
        """Parse LLM response to extract time estimates only (categories come from database)."""
        time_estimates = []
        lines = llm_response.split('\n')

        for i, place in enumerate(places):
            found_time = None

            for line in lines:
                if f"Place {i+1}:" in line or f"{i+1}." in line or place['name'] in line:
                    time_match = re.search(r'(\d+\.?\d*)\s*(?:hours?|h)', line, re.IGNORECASE)
                    if time_match:
                        found_time = float(time_match.group(1))
                    break

            if found_time is None:
                estimated_duration = place.get('estimatedVisitDuration', '2 hours')
                duration_match = re.search(r'(\d+\.?\d*)', estimated_duration)
                found_time = float(duration_match.group(1)) if duration_match else 2.0

            time_estimates.append(found_time)

        return time_estimates
    
    def process_places_with_llm(self, places: List[Dict], user_preferences: Dict) -> tuple[str, List[float]]:
        """Process places with LLM to get time estimates and response"""
        llm_prompt = self.get_llm_recommendation_prompt(places, user_preferences)
        llm_response = self.call_mistral_llm(llm_prompt)
        llm_time_estimates = self.parse_llm_time_estimates(llm_response, places)
        
        return llm_response, llm_time_estimates