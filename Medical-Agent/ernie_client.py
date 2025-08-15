import os
import base64
import json
from openai import OpenAI
from typing import List, Dict, Optional

class ErnieClient:
    def __init__(self, host: str = "0.0.0.0", port: str = "8180"):
        """
        Initialize the ERNIE client
        
        Args:
            host: Host address of the ERNIE service
            port: Port number of the ERNIE service
        """
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}/v1"
        self.client = OpenAI(
            api_key="null",
            base_url=self.base_url
        )
    
    def encode_image(self, image_path: str) -> str:
        """
        Convert image to base64 encoding
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64-encoded image string
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            raise Exception(f"Image encoding failed: {str(e)}")
    
    def chat_completion(self, messages: List[Dict], stream: bool = True) -> str:
        """
        Basic chat completion method
        
        Args:
            messages: List of messages
            stream: Whether to use streaming response
            
        Returns:
            Complete response text
        """
        try:
            completion = self.client.chat.completions.create(
                model="null",
                messages=messages,
                stream=stream
            )
            
            if stream:
                result = ""
                for chunk in completion:
                    if chunk.choices and chunk.choices[0].delta.content:
                        result += chunk.choices[0].delta.content
                return result
            else:
                return completion.choices[0].message.content
        except Exception as e:
            return f"Request failed: {str(e)}"
    
    def text_generation(self, prompt: str, system_prompt: str = None) -> str:
        """
        Pure text generation
        
        Args:
            prompt: User input prompt
            system_prompt: System prompt (optional)
            
        Returns:
            Generated text
        """
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        return self.chat_completion(messages)
    
    def analyze_image_and_text(self, text: str = None, image_path: str = None) -> str:
        """
        Analyze image and text (multimodal analysis)
        
        Args:
            text: Text description
            image_path: Image path
            
        Returns:
            Analysis result
        """
        if not text and not image_path:
            return "Please provide text or image for analysis"
        
        content = []
        
        # Add image content
        if image_path:
            try:
                base64_image = self.encode_image(image_path)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
            except Exception as e:
                return f"Image processing failed: {str(e)}"
        
        # Add text content
        if text:
            content.append({
                "type": "text",
                "text": text
            })
        
        messages = [{
            "role": "user",
            "content": content
        }]
        
        return self.chat_completion(messages)
    
    def medical_image_analysis(self, image_path: str, custom_prompt: str = None) -> str:
        """
        Professional medical image analysis
        
        Args:
            image_path: Path to the medical image
            custom_prompt: Custom analysis prompt
            
        Returns:
            Medical image analysis result
        """
        default_prompt = """Please act as a professional medical imaging analysis assistant and detailedly describe the visible symptom characteristics in this medical image, including:
1. Location and distribution of lesions
2. Morphological features (size, shape, boundaries, etc.)
3. Color and texture changes
4. Possible pathological features
5. Abnormal signs requiring attention

Please describe using professional yet understandable language."""
        
        prompt = custom_prompt if custom_prompt else default_prompt
        
        return self.analyze_image_and_text(text=prompt, image_path=image_path)
    
    def analyze_symptoms(self, text: str) -> List[str]:
        """
        Extract symptom keywords from text
        
        Args:
            text: Symptom description text
            
        Returns:
            List of symptom keywords
        """
        prompt = f"""Please extract key symptoms from the following medical description, with requirements:
1. Each symptom should be a concise medical term or descriptive word
2. The return format must be a Python list, e.g.: ["fever", "cough", "fatigue"]
3. Do not include any additional explanatory text
4. Extract at most 10 most important symptoms

Patient description: {text}

Please directly return the symptom list:"""
        
        result = self.text_generation(prompt)
        
        try:
            # Clean possible extra characters
            result = result.strip()
            # Remove possible code block markers
            if result.startswith("```") and result.endswith("```"):
                result = result[3:-3].strip()
            if result.startswith("python"):
                result = result[6:].strip()
            
            # Attempt to parse as Python list
            symptoms = eval(result)
            if isinstance(symptoms, list):
                return [str(symptom).strip() for symptom in symptoms if symptom]
        except Exception as e:
            # If parsing fails, try to extract keywords from text
            import re
            symptoms = re.findall(r'["\'](.*?)["\']', result)
            if symptoms:
                return symptoms[:10]  # Limit to 10 symptoms maximum
        
        # Default return
        return ["Symptom description is unclear"]
    
    def analyze_risk(self, symptoms: List[str], medical_info: Dict = None) -> Dict:
        """
        Analyze risk level
        
        Args:
            symptoms: List of symptoms
            medical_info: Relevant medical information
            
        Returns:
            Risk assessment result dictionary
        """
        if not symptoms:
            return {
                "risk_level": 1,
                "urgency": "Please describe symptoms in detail",
                "recommendations": ["Please provide more detailed symptom description for accurate assessment"]
            }
        
        medical_context = ""
        if medical_info and medical_info.get("documents"):
            medical_context = f"\nRelevant medical knowledge: {medical_info}"
        
        prompt = f"""Please act as a professional medical risk assessment expert and evaluate the patient's risk level based on the following information:

Symptom list: {', '.join(symptoms)}
{medical_context}

Please return the assessment result in Python dictionary format, including the following fields:
- risk_level: Integer from 1-5 (1=very low risk, 2=low risk, 3=medium risk, 4=high risk, 5=urgent)
- urgency: Medical consultation recommendation (e.g.: "Recommend observation", "Routine clinic visit", "Emergency visit", "Immediate medical attention")
- recommendations: List of recommendations (3-5 specific suggestions)

Example format:
{{
    "risk_level": 3,
    "urgency": "Recommend routine clinic visit",
    "recommendations": [
        "It is advisable to seek medical examination in a timely manner",
        "Rest appropriately and avoid overexertion", 
        "Keep the affected area clean",
        "Closely monitor symptom changes"
    ]
}}

Please directly return the dictionary:"""
        
        result = self.text_generation(prompt)
        
        try:
            # Clean format
            result = result.strip()
            if result.startswith("```") and result.endswith("```"):
                result = result[3:-3].strip()
            if result.startswith("python") or result.startswith("json"):
                result = result.split('\n', 1)[1].strip()
            
            # Attempt to parse JSON or Python dictionary
            try:
                return eval(result)
            except:
                return json.loads(result)
        except Exception as e:
            # Return default values if parsing fails
            return {
                "risk_level": 2,
                "urgency": "Recommend medical consultation",
                "recommendations": [
                    "It is advisable to consult a professional doctor",
                    "Monitor symptom changes closely",
                    "Maintain good living habits",
                    "Seek prompt medical attention if symptoms worsen"
                ]
            }
    
    def generate_treatment_plan(self, symptoms: List[str], medical_info: Dict = None) -> Dict:
        """
        Generate treatment recommendation plan
        
        Args:
            symptoms: List of symptoms
            medical_info: Relevant medical information
            
        Returns:
            Treatment plan dictionary
        """
        if not symptoms:
            return {
                "examinations": ["Please provide detailed symptom description first"],
                "medications": ["Medication needs to be determined based on specific symptoms"],
                "lifestyle": ["Maintain a healthy lifestyle"]
            }
        
        medical_context = ""
        if medical_info and medical_info.get("documents"):
            medical_context = f"\nReference medical information: {medical_info}"
        
        prompt = f"""Please act as a professional medical consultant and generate a treatment recommendation plan based on the following symptom information:

Patient symptoms: {', '.join(symptoms)}
{medical_context}

Please return the treatment plan in Python dictionary format, including the following fields:
- examinations: List of recommended examination items (3-5 items)
- medications: List of medication recommendations (Note: Only general recommendations can be given, not specific prescription drugs)
- lifestyle: List of lifestyle recommendations (4-6 items)

Example format:
{{
    "examinations": [
        "Complete blood count test",
        "Relevant specialist examinations",
        "Imaging examinations"
    ],
    "medications": [
        "Please take medication as prescribed by your doctor",
        "Symptomatic treatment may be considered",
        "Avoid self-medication"
    ],
    "lifestyle": [
        "Adequate rest and avoid overexertion",
        "Keep the affected area clean and hygienic",
        "Eat light meals and drink plenty of water",
        "Avoid irritating foods"
    ]
}}

Please directly return the dictionary:"""
        
        result = self.text_generation(prompt)
        
        try:
            # Clean format
            result = result.strip()
            if result.startswith("```") and result.endswith("```"):
                result = result[3:-3].strip()
            if result.startswith("python") or result.startswith("json"):
                result = result.split('\n', 1)[1].strip()
            
            # Attempt to parse
            try:
                return eval(result)
            except:
                return json.loads(result)
        except Exception as e:
            # Return default plan if parsing fails
            return {
                "examinations": [
                    "Recommended relevant specialist examinations",
                    "Basic physical examination items",
                    "Necessary imaging examinations"
                ],
                "medications": [
                    "Please follow the medication guidance of professional doctors",
                    "Avoid purchasing and using prescription drugs without authorization",
                    "Symptomatic drugs may be used under doctor's guidance"
                ],
                "lifestyle": [
                    "Maintain adequate rest and sleep",
                    "Maintain a nutritionally balanced diet and avoid irritating foods",
                    "Exercise appropriately to enhance physical fitness",
                    "Maintain good personal hygiene",
                    "Regular review and monitor symptom changes"
                ]
            }
    
    def test_connection(self) -> bool:
        """
        Test connection to ERNIE service
        
        Returns:
            Whether the connection is successful
        """
        try:
            test_response = self.text_generation("Hello")
            return len(test_response) > 0 and "Request failed" not in test_response
        except:
            return False