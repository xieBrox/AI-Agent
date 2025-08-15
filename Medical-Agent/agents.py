from typing import List, Dict
import json
import logging
import os
from datetime import datetime
from knowledge_base import KnowledgeBase
from ernie_client import ErnieClient

# Configure a dedicated logger for agents
def setup_agents_logging():
    """Configure the logger for the agents module"""
    debug_mode = os.getenv('DEBUG', 'false').lower() in ['true', '1', 'yes', 'on']
    log_level = logging.DEBUG if debug_mode else logging.INFO
    
    agents_logger = logging.getLogger('agents_medical_app')
    agents_logger.setLevel(log_level)
    
    # Avoid adding duplicate handlers
    if not agents_logger.handlers:
        # File handler
        os.makedirs("logs", exist_ok=True)
        
        if debug_mode:
            log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s'
        else:
            log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        file_handler = logging.FileHandler(
            f"logs/agents_{datetime.now().strftime('%Y%m%d')}.log", 
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(log_format, '%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        agents_logger.addHandler(file_handler)
        
        if debug_mode:
            agents_logger.debug("🔧 Agents module DEBUG mode enabled")
    
    return agents_logger

# Initialize agents logger
agents_logger = setup_agents_logging()

class SymptomParserAgent:
    def __init__(self, ernie_client: ErnieClient):
        agents_logger.debug("Initializing SymptomParserAgent")
        self.ernie = ernie_client
        agents_logger.info("SymptomParserAgent initialization completed")
    
    def parse_symptoms(self, text_input: str, image_analysis: str = None) -> List[str]:
        """Parse symptom text and image analysis results to extract key symptoms"""
        parse_start_time = datetime.now()
        parser_id = parse_start_time.strftime('%H%M%S_%f')[:9]
        
        agents_logger.info(f"[Parser-{parser_id}] Starting symptom parsing")
        agents_logger.debug(f"[Parser-{parser_id}] Input parameters:")
        agents_logger.debug(f"[Parser-{parser_id}] - Text input length: {len(text_input or '')}")
        agents_logger.debug(f"[Parser-{parser_id}] - Image analysis: {'Available' if image_analysis else 'None'}")
        
        combined_text = text_input or ""
        if image_analysis:
            combined_text = f"{combined_text}\nImage analysis results: {image_analysis}"
            agents_logger.debug(f"[Parser-{parser_id}] Merged image analysis results, final text length: {len(combined_text)}")
        
        if not combined_text.strip():
            agents_logger.warning(f"[Parser-{parser_id}] Input text is empty, returning empty symptom list")
            return []
        
        agents_logger.debug(f"[Parser-{parser_id}] Calling ERNIE for symptom analysis")
        agents_logger.debug(f"[Parser-{parser_id}] Analysis text: {repr(combined_text[:200])}")
        
        try:
            symptoms = self.ernie.analyze_symptoms(combined_text)
            parse_time = (datetime.now() - parse_start_time).total_seconds()
            
            agents_logger.info(f"[Parser-{parser_id}] Symptom parsing completed")
            agents_logger.debug(f"[Parser-{parser_id}] Parsing results: {symptoms}")
            agents_logger.debug(f"[Parser-{parser_id}] Parsing duration: {parse_time:.3f} seconds")
            agents_logger.debug(f"[Parser-{parser_id}] Number of identified symptoms: {len(symptoms)}")
            
            return symptoms
            
        except Exception as e:
            error_time = (datetime.now() - parse_start_time).total_seconds()
            agents_logger.error(f"[Parser-{parser_id}] Symptom parsing failed: {str(e)}")
            agents_logger.debug(f"[Parser-{parser_id}] Error occurred after: {error_time:.3f} seconds")
            agents_logger.debug(f"[Parser-{parser_id}] Error details: {repr(e)}", exc_info=True)
            return []

class KnowledgeRetrievalAgent:
    def __init__(self, knowledge_base: KnowledgeBase):
        agents_logger.debug("Initializing KnowledgeRetrievalAgent")
        self.kb = knowledge_base
        agents_logger.info("KnowledgeRetrievalAgent initialization completed")
    
    def retrieve_relevant_info(self, symptoms: List[str]) -> Dict:
        """Retrieve relevant medical knowledge"""
        retrieve_start_time = datetime.now()
        retriever_id = retrieve_start_time.strftime('%H%M%S_%f')[:9]
        
        agents_logger.info(f"[Retriever-{retriever_id}] Starting medical knowledge retrieval")
        agents_logger.debug(f"[Retriever-{retriever_id}] Input symptoms: {symptoms}")
        agents_logger.debug(f"[Retriever-{retriever_id}] Number of symptoms: {len(symptoms)}")
        
        if not symptoms:
            agents_logger.warning(f"[Retriever-{retriever_id}] Symptom list is empty, returning default structure")
            default_result = {
                "diseases": {"documents": [[]]},
                "treatments": {}
            }
            agents_logger.debug(f"[Retriever-{retriever_id}] Returning default result: {default_result}")
            return default_result
        
        try:
            # Get disease information
            agents_logger.debug(f"[Retriever-{retriever_id}] Querying disease information...")
            disease_start = datetime.now()
            disease_info = self.kb.get_disease_info(symptoms)
            disease_time = (datetime.now() - disease_start).total_seconds()
            
            agents_logger.debug(f"[Retriever-{retriever_id}] Disease query completed, duration: {disease_time:.3f} seconds")
            agents_logger.debug(f"[Retriever-{retriever_id}] Disease info structure: {list(disease_info.keys()) if disease_info else 'None'}")
            
            # Get treatment recommendations for each disease
            treatment_info = {}
            if disease_info.get("documents") and disease_info["documents"][0]:
                diseases = disease_info["documents"][0]
                agents_logger.debug(f"[Retriever-{retriever_id}] Found {len(diseases)} related diseases: {diseases}")
                
                for i, disease in enumerate(diseases):
                    agents_logger.debug(f"[Retriever-{retriever_id}] Querying disease {i+1}/{len(diseases)}: {disease}")
                    treatment_start = datetime.now()
                    
                    try:
                        treatment_info[disease] = self.kb.get_treatment_suggestions(disease)
                        treatment_time = (datetime.now() - treatment_start).total_seconds()
                        agents_logger.debug(f"[Retriever-{retriever_id}] Treatment recommendations for {disease} obtained, duration: {treatment_time:.3f} seconds")
                    except Exception as e:
                        agents_logger.error(f"[Retriever-{retriever_id}] Failed to get treatment recommendations for {disease}: {str(e)}")
                        treatment_info[disease] = None
            else:
                agents_logger.debug(f"[Retriever-{retriever_id}] No relevant disease information found")
            
            result = {
                "diseases": disease_info,
                "treatments": treatment_info
            }
            
            total_time = (datetime.now() - retrieve_start_time).total_seconds()
            agents_logger.info(f"[Retriever-{retriever_id}] Medical knowledge retrieval completed")
            agents_logger.debug(f"[Retriever-{retriever_id}] Retrieval statistics:")
            agents_logger.debug(f"[Retriever-{retriever_id}] - Total duration: {total_time:.3f} seconds")
            agents_logger.debug(f"[Retriever-{retriever_id}] - Number of related diseases: {len(treatment_info)}")
            agents_logger.debug(f"[Retriever-{retriever_id}] - Number of treatment recommendations: {sum(1 for v in treatment_info.values() if v)}")
            
            return result
            
        except Exception as e:
            error_time = (datetime.now() - retrieve_start_time).total_seconds()
            agents_logger.error(f"[Retriever-{retriever_id}] Medical knowledge retrieval failed: {str(e)}")
            agents_logger.debug(f"[Retriever-{retriever_id}] Error occurred after: {error_time:.3f} seconds")
            agents_logger.debug(f"[Retriever-{retriever_id}] Error details: {repr(e)}", exc_info=True)
            
            # Return empty result instead of crashing
            return {
                "diseases": {"documents": [[]]},
                "treatments": {}
            }

class DiagnosisAgent:
    def __init__(self, ernie_client: ErnieClient):
        agents_logger.debug("Initializing DiagnosisAgent")
        self.ernie = ernie_client
        agents_logger.info("DiagnosisAgent initialization completed")
    
    def analyze_risk_level(self, symptoms: List[str], medical_info: Dict) -> Dict:
        """Analyze risk level and generate recommendations"""
        risk_start_time = datetime.now()
        risk_id = risk_start_time.strftime('%H%M%S_%f')[:9]
        
        agents_logger.info(f"[Risk-{risk_id}] Starting risk assessment")
        agents_logger.debug(f"[Risk-{risk_id}] Input symptoms: {symptoms}")
        agents_logger.debug(f"[Risk-{risk_id}] Medical info structure: {list(medical_info.keys()) if medical_info else 'None'}")
        
        try:
            risk_result = self.ernie.analyze_risk(symptoms, medical_info)
            risk_time = (datetime.now() - risk_start_time).total_seconds()
            
            agents_logger.info(f"[Risk-{risk_id}] Risk assessment completed")
            agents_logger.debug(f"[Risk-{risk_id}] Assessment duration: {risk_time:.3f} seconds")
            agents_logger.debug(f"[Risk-{risk_id}] Risk assessment results: {risk_result}")
            
            return risk_result
            
        except Exception as e:
            error_time = (datetime.now() - risk_start_time).total_seconds()
            agents_logger.error(f"[Risk-{risk_id}] Risk assessment failed: {str(e)}")
            agents_logger.debug(f"[Risk-{risk_id}] Error occurred after: {error_time:.3f} seconds")
            agents_logger.debug(f"[Risk-{risk_id}] Error details: {repr(e)}", exc_info=True)
            
            # Return default risk assessment
            return {
                "risk_level": 1,
                "urgency": "Recommend consulting a doctor",
                "recommendations": ["Please describe symptoms in detail", "Recommend seeking medical attention promptly"]
            }
    
    def generate_treatment_plan(self, symptoms: List[str], medical_info: Dict) -> Dict:
        """Generate treatment plan recommendations"""
        plan_start_time = datetime.now()
        plan_id = plan_start_time.strftime('%H%M%S_%f')[:9]
        
        agents_logger.info(f"[Plan-{plan_id}] Starting treatment plan generation")
        agents_logger.debug(f"[Plan-{plan_id}] Input symptoms: {symptoms}")
        agents_logger.debug(f"[Plan-{plan_id}] Medical info structure: {list(medical_info.keys()) if medical_info else 'None'}")
        
        try:
            plan_result = self.ernie.generate_treatment_plan(symptoms, medical_info)
            plan_time = (datetime.now() - plan_start_time).total_seconds()
            
            agents_logger.info(f"[Plan-{plan_id}] Treatment plan generation completed")
            agents_logger.debug(f"[Plan-{plan_id}] Generation duration: {plan_time:.3f} seconds")
            agents_logger.debug(f"[Plan-{plan_id}] Treatment plan results: {plan_result}")
            
            if isinstance(plan_result, dict):
                agents_logger.debug(f"[Plan-{plan_id}] Plan includes:")
                for key, value in plan_result.items():
                    if isinstance(value, list):
                        agents_logger.debug(f"[Plan-{plan_id}] - {key}: {len(value)} recommendations")
                    else:
                        agents_logger.debug(f"[Plan-{plan_id}] - {key}: {type(value).__name__}")
            
            return plan_result
            
        except Exception as e:
            error_time = (datetime.now() - plan_start_time).total_seconds()
            agents_logger.error(f"[Plan-{plan_id}] Treatment plan generation failed: {str(e)}")
            agents_logger.debug(f"[Plan-{plan_id}] Error occurred after: {error_time:.3f} seconds")
            agents_logger.debug(f"[Plan-{plan_id}] Error details: {repr(e)}", exc_info=True)
            
            # Return default treatment plan
            return {
                "examinations": ["Routine physical examination", "Relevant tests if necessary"],
                "medications": ["Please take medication as prescribed by your doctor"],
                "lifestyle": ["Get adequate rest", "Maintain good living habits", "Seek medical attention promptly"]
            }

class AgentCoordinator:
    def __init__(self):
        coord_init_time = datetime.now()
        agents_logger.info("Starting AgentCoordinator initialization")
        
        try:
            # Initialize core clients
            agents_logger.debug("Initializing ERNIE client...")
            ernie_start = datetime.now()
            self.ernie = ErnieClient()
            ernie_time = (datetime.now() - ernie_start).total_seconds()
            agents_logger.debug(f"ERNIE client initialization completed, duration: {ernie_time:.3f} seconds")
            
            # Initialize knowledge base
            agents_logger.debug("Initializing knowledge base...")
            kb_start = datetime.now()
            self.kb = KnowledgeBase()
            kb_time = (datetime.now() - kb_start).total_seconds()
            agents_logger.debug(f"Knowledge base initialization completed, duration: {kb_time:.3f} seconds")
            
            # Initialize individual agents
            agents_logger.debug("Initializing symptom parsing agent...")
            self.symptom_parser = SymptomParserAgent(self.ernie)
            
            agents_logger.debug("Initializing knowledge retrieval agent...")
            self.knowledge_retriever = KnowledgeRetrievalAgent(self.kb)
            
            agents_logger.debug("Initializing diagnosis agent...")
            self.diagnosis_agent = DiagnosisAgent(self.ernie)
            
            total_init_time = (datetime.now() - coord_init_time).total_seconds()
            agents_logger.info(f"AgentCoordinator initialization completed, total duration: {total_init_time:.3f} seconds")
            
        except Exception as e:
            init_error_time = (datetime.now() - coord_init_time).total_seconds()
            agents_logger.error(f"AgentCoordinator initialization failed: {str(e)}")
            agents_logger.debug(f"Initialization failed after: {init_error_time:.3f} seconds")
            agents_logger.debug(f"Initialization error details: {repr(e)}", exc_info=True)
            raise
    
    def process_consultation(self, text_input: str, image_path: str = None) -> Dict:
        """Coordinate multiple agents to complete the consultation process"""
        consultation_start_time = datetime.now()
        coord_id = consultation_start_time.strftime('%H%M%S_%f')[:9]
        
        agents_logger.info(f"[Coord-{coord_id}] Starting consultation process coordination")
        agents_logger.debug(f"[Coord-{coord_id}] Input parameters:")
        agents_logger.debug(f"[Coord-{coord_id}] - Text input: {repr(text_input[:100]) if text_input else None}")
        agents_logger.debug(f"[Coord-{coord_id}] - Image path: {image_path}")
        
        try:
            # 1. Process image input
            image_analysis = None
            if image_path:
                agents_logger.debug(f"[Coord-{coord_id}] Step 1: Processing image input")
                image_start = datetime.now()
                
                try:
                    image_analysis = self.ernie.medical_image_analysis(image_path)
                    image_time = (datetime.now() - image_start).total_seconds()
                    agents_logger.debug(f"[Coord-{coord_id}] Image analysis successful, duration: {image_time:.3f} seconds")
                    agents_logger.debug(f"[Coord-{coord_id}] Image analysis result length: {len(str(image_analysis))}")
                except Exception as e:
                    image_analysis = f"Image processing error: {str(e)}"
                    image_error_time = (datetime.now() - image_start).total_seconds()
                    agents_logger.error(f"[Coord-{coord_id}] Image analysis failed: {str(e)}")
                    agents_logger.debug(f"[Coord-{coord_id}] Image analysis error occurred after: {image_error_time:.3f} seconds")
            else:
                agents_logger.debug(f"[Coord-{coord_id}] Step 1: No image input, skipping image processing")
            
            # 2. Parse symptoms
            agents_logger.debug(f"[Coord-{coord_id}] Step 2: Parsing symptoms")
            symptom_start = datetime.now()
            symptoms = self.symptom_parser.parse_symptoms(text_input, image_analysis)
            symptom_time = (datetime.now() - symptom_start).total_seconds()
            agents_logger.debug(f"[Coord-{coord_id}] Symptom parsing completed, duration: {symptom_time:.3f} seconds")
            agents_logger.debug(f"[Coord-{coord_id}] Parsed symptoms: {symptoms}")
            
            # 3. Retrieve relevant medical knowledge
            agents_logger.debug(f"[Coord-{coord_id}] Step 3: Retrieving medical knowledge")
            knowledge_start = datetime.now()
            medical_info = self.knowledge_retriever.retrieve_relevant_info(symptoms)
            knowledge_time = (datetime.now() - knowledge_start).total_seconds()
            agents_logger.debug(f"[Coord-{coord_id}] Medical knowledge retrieval completed, duration: {knowledge_time:.3f} seconds")
            
            # 4. Risk assessment and treatment planning
            agents_logger.debug(f"[Coord-{coord_id}] Step 4: Performing risk assessment")
            risk_start = datetime.now()
            risk_assessment = self.diagnosis_agent.analyze_risk_level(symptoms, medical_info)
            risk_time = (datetime.now() - risk_start).total_seconds()
            agents_logger.debug(f"[Coord-{coord_id}] Risk assessment completed, duration: {risk_time:.3f} seconds")
            
            agents_logger.debug(f"[Coord-{coord_id}] Step 5: Generating treatment plan")
            treatment_start = datetime.now()
            treatment_plan = self.diagnosis_agent.generate_treatment_plan(symptoms, medical_info)
            treatment_time = (datetime.now() - treatment_start).total_seconds()
            agents_logger.debug(f"[Coord-{coord_id}] Treatment plan generation completed, duration: {treatment_time:.3f} seconds")
            
            # Construct final result
            result = {
                "symptoms": symptoms,
                "medical_info": medical_info,
                "risk_assessment": risk_assessment,
                "treatment_plan": treatment_plan,
                "image_analysis": image_analysis
            }
            
            total_time = (datetime.now() - consultation_start_time).total_seconds()
            agents_logger.info(f"[Coord-{coord_id}] Consultation process coordination completed")
            agents_logger.debug(f"[Coord-{coord_id}] Performance statistics:")
            agents_logger.debug(f"[Coord-{coord_id}] - Image processing: {(datetime.now() - image_start).total_seconds():.3f} seconds" if image_path else " - Image processing: Skipped")
            agents_logger.debug(f"[Coord-{coord_id}] - Symptom parsing: {symptom_time:.3f} seconds")
            agents_logger.debug(f"[Coord-{coord_id}] - Knowledge retrieval: {knowledge_time:.3f} seconds")
            agents_logger.debug(f"[Coord-{coord_id}] - Risk assessment: {risk_time:.3f} seconds")
            agents_logger.debug(f"[Coord-{coord_id}] - Treatment planning: {treatment_time:.3f} seconds")
            agents_logger.debug(f"[Coord-{coord_id}] - Total duration: {total_time:.3f} seconds")
            agents_logger.debug(f"[Coord-{coord_id}] Final result includes: {list(result.keys())}")
            
            return result
            
        except Exception as e:
            error_time = (datetime.now() - consultation_start_time).total_seconds()
            agents_logger.error(f"[Coord-{coord_id}] Consultation process coordination failed: {str(e)}")
            agents_logger.debug(f"[Coord-{coord_id}] Error occurred after: {error_time:.3f} seconds")
            agents_logger.debug(f"[Coord-{coord_id}] Error details: {repr(e)}", exc_info=True)
            
            # Return error result instead of crashing
            return {
                "symptoms": [],
                "medical_info": {"diseases": {"documents": [[]]}, "treatments": {}},
                "risk_assessment": {"risk_level": 0, "urgency": "System error", "recommendations": ["System processing error, please try again"]},
                "treatment_plan": {"examinations": [], "medications": [], "lifestyle": []},
                "image_analysis": None
            }
    
    def test_system(self) -> Dict:
        """Test whether all system components are functioning properly"""
        test_start_time = datetime.now()
        test_id = test_start_time.strftime('%H%M%S_%f')[:9]
        
        agents_logger.info(f"[Test-{test_id}] Starting system self-test")
        test_results = {}
        
        # Test ERNIE connection
        agents_logger.debug(f"[Test-{test_id}] Test 1: ERNIE connection")
        ernie_test_start = datetime.now()
        try:
            test_results["ernie_connection"] = self.ernie.test_connection()
            ernie_test_time = (datetime.now() - ernie_test_start).total_seconds()
            agents_logger.debug(f"[Test-{test_id}] ERNIE connection test completed, duration: {ernie_test_time:.3f} seconds, result: {test_results['ernie_connection']}")
        except Exception as e:
            test_results["ernie_connection"] = False
            test_results["ernie_connection_error"] = str(e)
            agents_logger.error(f"[Test-{test_id}] ERNIE connection test failed: {str(e)}")
        
        # Test knowledge base
        agents_logger.debug(f"[Test-{test_id}] Test 2: Knowledge base")
        kb_test_start = datetime.now()
        try:
            test_symptoms = ["fever", "cough"]
            agents_logger.debug(f"[Test-{test_id}] Using test symptoms: {test_symptoms}")
            medical_info = self.knowledge_retriever.retrieve_relevant_info(test_symptoms)
            kb_test_time = (datetime.now() - kb_test_start).total_seconds()
            
            diseases_found = len(medical_info.get("diseases", {}).get("documents", [[]])[0]) > 0
            test_results["knowledge_base"] = diseases_found
            
            agents_logger.debug(f"[Test-{test_id}] Knowledge base test completed, duration: {kb_test_time:.3f} seconds")
            agents_logger.debug(f"[Test-{test_id}] Knowledge base result: {diseases_found}")
            agents_logger.debug(f"[Test-{test_id}] Number of diseases found: {len(medical_info.get('diseases', {}).get('documents', [[]])[0])}")
            
        except Exception as e:
            test_results["knowledge_base"] = False
            test_results["knowledge_base_error"] = str(e)
            kb_error_time = (datetime.now() - kb_test_start).total_seconds()
            agents_logger.error(f"[Test-{test_id}] Knowledge base test failed: {str(e)}")
            agents_logger.debug(f"[Test-{test_id}] Knowledge base error occurred after: {kb_error_time:.3f} seconds")
        
        # Test symptom parsing
        agents_logger.debug(f"[Test-{test_id}] Test 3: Symptom parsing")
        symptom_test_start = datetime.now()
        try:
            test_text = "I've been experiencing fever and cough lately"
            agents_logger.debug(f"[Test-{test_id}] Using test text: {repr(test_text)}")
            symptoms = self.symptom_parser.parse_symptoms(test_text)
            symptom_test_time = (datetime.now() - symptom_test_start).total_seconds()
            
            parsing_success = len(symptoms) > 0
            test_results["symptom_parsing"] = parsing_success
            
            agents_logger.debug(f"[Test-{test_id}] Symptom parsing test completed, duration: {symptom_test_time:.3f} seconds")
            agents_logger.debug(f"[Test-{test_id}] Symptom parsing result: {parsing_success}")
            agents_logger.debug(f"[Test-{test_id}] Parsed symptoms: {symptoms}")
            
        except Exception as e:
            test_results["symptom_parsing"] = False
            test_results["symptom_parsing_error"] = str(e)
            symptom_error_time = (datetime.now() - symptom_test_start).total_seconds()
            agents_logger.error(f"[Test-{test_id}] Symptom parsing test failed: {str(e)}")
            agents_logger.debug(f"[Test-{test_id}] Symptom parsing error occurred after: {symptom_error_time:.3f} seconds")
        
        total_test_time = (datetime.now() - test_start_time).total_seconds()
        
        # Calculate test results
        passed_tests = sum(1 for k, v in test_results.items() if not k.endswith('_error') and v)
        total_tests = len([k for k in test_results.keys() if not k.endswith('_error')])
        
        agents_logger.info(f"[Test-{test_id}] System self-test completed")
        agents_logger.debug(f"[Test-{test_id}] Self-test statistics:")
        agents_logger.debug(f"[Test-{test_id}] - Total duration: {total_test_time:.3f} seconds")
        agents_logger.debug(f"[Test-{test_id}] - Passed tests: {passed_tests}/{total_tests}")
        agents_logger.debug(f"[Test-{test_id}] - Test results: {test_results}")
        
        return test_results