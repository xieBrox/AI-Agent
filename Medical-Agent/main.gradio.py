import os
import logging
from datetime import datetime
import gradio as gr
from dotenv import load_dotenv
from agents import AgentCoordinator

# Load environment variables
load_dotenv()

# Configure Gradio logging
def setup_gradio_logging():
    """Configure Gradio-specific logger - supports DEBUG mode"""
    # Create log directory
    os.makedirs("logs", exist_ok=True)
    
    # Check if DEBUG mode is enabled
    debug_mode = os.getenv('DEBUG', 'false').lower() in ['true', '1', 'yes', 'on']
    log_level = logging.DEBUG if debug_mode else logging.INFO
    
    # Configure log format
    if debug_mode:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s'
    else:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Create Gradio-specific logger
    gradio_logger = logging.getLogger('gradio_medical_app')
    gradio_logger.setLevel(log_level)
    
    # Clear existing handlers
    gradio_logger.handlers.clear()
    
    # File handler - records all logs to file
    file_handler = logging.FileHandler(
        f"logs/gradio_app_{datetime.now().strftime('%Y%m%d')}.log", 
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(log_format, date_format)
    file_handler.setFormatter(file_formatter)
    
    # DEBUG file handler - separate debug log file
    if debug_mode:
        debug_handler = logging.FileHandler(
            f"logs/debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", 
            encoding='utf-8'
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(file_formatter)
        gradio_logger.addHandler(debug_handler)
    
    # Console handler - more details in DEBUG mode
    console_handler = logging.StreamHandler()
    if debug_mode:
        console_handler.setLevel(logging.DEBUG)
        console_formatter = logging.Formatter('%(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    else:
        console_handler.setLevel(logging.WARNING)
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    gradio_logger.addHandler(file_handler)
    gradio_logger.addHandler(console_handler)
    
    # Log DEBUG mode status
    if debug_mode:
        gradio_logger.info("🔧 DEBUG mode enabled - detailed logging active...")
        gradio_logger.debug(f"Log level: {logging.getLevelName(log_level)}")
        gradio_logger.debug(f"Log format: {log_format}")
    else:
        gradio_logger.info("ℹ️ Running in standard mode - set DEBUG=true environment variable to enable debugging")
    
    return gradio_logger

# Initialize logger
logger = setup_gradio_logging()

class MedicalConsultation:
    def __init__(self):
        logger.info("Initializing intelligent medical consultation system")
        logger.debug("Creating MedicalConsultation instance")
        
        try:
            logger.debug("Initializing AgentCoordinator...")
            start_time = datetime.now()
            self.coordinator = AgentCoordinator()
            init_time = (datetime.now() - start_time).total_seconds()
            
            logger.info("AgentCoordinator initialized successfully")
            logger.debug(f"AgentCoordinator initialization time: {init_time:.3f} seconds")
            
            # Verify critical components of coordinator
            logger.debug("Verifying AgentCoordinator components...")
            if hasattr(self.coordinator, 'ernie_client'):
                logger.debug("✅ ERNIE client component loaded")
            if hasattr(self.coordinator, 'knowledge_base'):
                logger.debug("✅ Knowledge base component loaded")
                
        except Exception as e:
            logger.error(f"AgentCoordinator initialization failed: {str(e)}")
            logger.debug(f"Detailed error information for initialization failure: {repr(e)}", exc_info=True)
            raise

    def format_results(self, consultation_results: dict) -> str:
        """Format diagnosis results"""
        format_start_time = datetime.now()
        logger.debug("Starting to format diagnosis results")
        logger.debug(f"Input data structure: {type(consultation_results)}")
        
        if not consultation_results:
            logger.warning("consultation_results is empty, cannot format")
            return "⚠️ No valid diagnosis results received"
        
        logger.debug(f"Keys in consultation_results: {list(consultation_results.keys())}")
        
        output = []
        sections_added = 0
        
        # Add image analysis results
        image_analysis = consultation_results.get("image_analysis")
        if image_analysis:
            logger.debug(f"Adding image analysis results (length: {len(str(image_analysis))})")
            output.append("【Image Analysis】")
            output.append(image_analysis)
            output.append("")
            sections_added += 1
            logger.info("Added image analysis results to output")
        else:
            logger.debug("No image analysis results")
        
        # Add symptom analysis
        symptoms = consultation_results.get("symptoms", [])
        logger.debug(f"Symptom analysis data: {symptoms} (type: {type(symptoms)})")
        
        output.append("【Symptom Analysis】")
        if isinstance(symptoms, list) and symptoms:
            symptoms_text = ", ".join(str(s) for s in symptoms)
            output.append("Identified symptoms: " + symptoms_text)
            logger.debug(f"Symptom text: {symptoms_text}")
        else:
            output.append("Identified symptoms: No specific symptoms")
            logger.debug("Symptom list is empty or incorrectly formatted")
        output.append("")
        sections_added += 1
        logger.info(f"Identified {len(symptoms)} symptoms: {symptoms}")
        
        # Add risk assessment
        risk = consultation_results.get("risk_assessment", {})
        logger.debug(f"Risk assessment data: {risk} (type: {type(risk)})")
        
        risk_level = risk.get("risk_level", 0) if isinstance(risk, dict) else 0
        urgency = risk.get("urgency", "Unknown") if isinstance(risk, dict) else "Unknown"
        recommendations = risk.get("recommendations", []) if isinstance(risk, dict) else []
        
        logger.debug(f"Risk level: {risk_level}, Urgency: {urgency}, Number of recommendations: {len(recommendations)}")
        
        output.append("【Risk Assessment】")
        output.append(f"Risk level: {'⚠️' * int(risk_level)} ({urgency})")
        output.append("Recommendations:")
        
        if isinstance(recommendations, list):
            for i, rec in enumerate(recommendations):
                output.append(f"- {rec}")
                logger.debug(f"Recommendation {i+1}: {rec}")
        else:
            output.append("- No specific recommendations")
            logger.debug("recommendations is not in list format")
            
        output.append("")
        sections_added += 1
        logger.info(f"Risk assessment completed - Level: {risk_level}, Recommendation: {urgency}")
        
        # Add treatment plan
        plan = consultation_results.get("treatment_plan", {})
        logger.debug(f"Treatment plan data: {plan} (type: {type(plan)})")
        
        if isinstance(plan, dict):
            examinations = plan.get("examinations", [])
            medications = plan.get("medications", [])
            lifestyle = plan.get("lifestyle", [])
        else:
            examinations = medications = lifestyle = []
            logger.debug("treatment_plan is not in dictionary format")
        
        logger.debug(f"Number of recommended examinations: {len(examinations)}")
        logger.debug(f"Number of medication recommendations: {len(medications)}")
        logger.debug(f"Number of lifestyle recommendations: {len(lifestyle)}")
        
        # Recommended examinations
        output.append("【Recommended Examinations】")
        if examinations:
            for i, exam in enumerate(examinations):
                output.append(f"- {exam}")
                logger.debug(f"Examination {i+1}: {exam}")
        else:
            output.append("- No special examination recommendations")
        output.append("")
        sections_added += 1
        
        # Medication recommendations
        output.append("【Medication Recommendations】")
        if medications:
            for i, med in enumerate(medications):
                output.append(f"- {med}")
                logger.debug(f"Medication recommendation {i+1}: {med}")
        else:
            output.append("- Please follow doctor's prescription")
        output.append("")
        sections_added += 1
        
        # Lifestyle recommendations
        output.append("【Lifestyle Recommendations】")
        if lifestyle:
            for i, advice in enumerate(lifestyle):
                output.append(f"- {advice}")
                logger.debug(f"Lifestyle recommendation {i+1}: {advice}")
        else:
            output.append("- Get adequate rest and maintain healthy habits")
        sections_added += 1
        
        format_time = (datetime.now() - format_start_time).total_seconds()
        result_text = "\n".join(output)
        
        logger.info(f"Treatment plan generated - Examinations: {len(examinations)}, Medications: {len(medications)}, Lifestyle advice: {len(lifestyle)}")
        logger.debug(f"Formatting statistics:")
        logger.debug(f"- Added {sections_added} result sections")
        logger.debug(f"- Formatting time: {format_time:.3f} seconds")
        logger.debug(f"- Final result length: {len(result_text)} characters")
        logger.debug(f"- Result lines: {len(output)} lines")
        logger.debug("Diagnosis results formatting completed")
            
        return result_text

    def process_consultation(self, image, symptoms: str) -> str:
        """Process consultation request"""
        processing_start_time = datetime.now()
        consultation_id = processing_start_time.strftime('%Y%m%d_%H%M%S_%f')[:17]
        
        # Log user request
        user_info = {
            "consultation_id": consultation_id,
            "has_image": image is not None,
            "image_path": image if image else None,
            "symptoms_length": len(symptoms) if symptoms else 0,
            "symptoms_preview": symptoms[:100] + "..." if symptoms and len(symptoms) > 100 else symptoms,
            "timestamp": processing_start_time.isoformat()
        }
        logger.info(f"Received user consultation request: {user_info}")
        logger.debug(f"[{consultation_id}] Detailed request information:")
        logger.debug(f"[{consultation_id}] - Image: {'Yes' if image else 'No'}")
        logger.debug(f"[{consultation_id}] - Symptom text: {repr(symptoms)}")
        
        # Input validation
        if not symptoms and not image:
            logger.warning(f"[{consultation_id}] User provided neither symptom description nor image")
            logger.debug(f"[{consultation_id}] Input validation failed - no valid input")
            return "Please provide symptom description or upload an image"
        
        try:
            logger.info(f"[{consultation_id}] Starting to process consultation request")
            logger.debug(f"[{consultation_id}] Calling coordinator.process_consultation...")
            
            # Log call parameters
            call_params = {
                "text_input": symptoms or "",
                "image_path": image,
                "text_length": len(symptoms or ""),
                "has_image": image is not None
            }
            logger.debug(f"[{consultation_id}] Call parameters: {call_params}")
            
            # Process consultation using Agent coordinator
            coordination_start = datetime.now()
            consultation_results = self.coordinator.process_consultation(
                text_input=symptoms or "",
                image_path=image
            )
            coordination_time = (datetime.now() - coordination_start).total_seconds()
            
            logger.info(f"[{consultation_id}] Consultation processing completed, starting result formatting")
            logger.debug(f"[{consultation_id}] Agent coordinator processing time: {coordination_time:.3f} seconds")
            logger.debug(f"[{consultation_id}] Coordinator returned result structure: {list(consultation_results.keys()) if consultation_results else 'None'}")
            
            # Detailed logging of returned results
            if consultation_results:
                logger.debug(f"[{consultation_id}] Result details:")
                for key, value in consultation_results.items():
                    if isinstance(value, (str, int, float)):
                        logger.debug(f"[{consultation_id}] - {key}: {value}")
                    elif isinstance(value, (list, dict)):
                        logger.debug(f"[{consultation_id}] - {key}: {type(value).__name__}(length={len(value)})")
                    else:
                        logger.debug(f"[{consultation_id}] - {key}: {type(value).__name__}")
            
            # Format output results
            formatting_start = datetime.now()
            formatted_result = self.format_results(consultation_results)
            formatting_time = (datetime.now() - formatting_start).total_seconds()
            
            total_time = (datetime.now() - processing_start_time).total_seconds()
            
            logger.info(f"[{consultation_id}] Consultation request processed successfully")
            logger.debug(f"[{consultation_id}] Performance statistics:")
            logger.debug(f"[{consultation_id}] - Coordinator processing: {coordination_time:.3f} seconds")
            logger.debug(f"[{consultation_id}] - Result formatting: {formatting_time:.3f} seconds")
            logger.debug(f"[{consultation_id}] - Total time: {total_time:.3f} seconds")
            logger.debug(f"[{consultation_id}] - Result length: {len(formatted_result)} characters")
            
            return formatted_result
            
        except Exception as e:
            error_time = (datetime.now() - processing_start_time).total_seconds()
            error_msg = f"System processing error: {str(e)}"
            
            logger.error(f"[{consultation_id}] Consultation processing failed: {str(e)}")
            logger.debug(f"[{consultation_id}] Error occurred after: {error_time:.3f} seconds")
            logger.debug(f"[{consultation_id}] Error type: {type(e).__name__}")
            logger.debug(f"[{consultation_id}] Error details: {repr(e)}", exc_info=True)
            
            return f"{error_msg}\n\nPlease check:\n1. Is the ERNIE service running properly (http://0.0.0.0:8180)\n2. Has the knowledge base been initialized\n3. Is the network connection working"

def create_ui():
    """Create Gradio interface"""
    logger.info("Starting to create Gradio user interface")
    
    try:
        consultation = MedicalConsultation()
        logger.info("MedicalConsultation instance created successfully")
    except Exception as e:
        logger.error(f"Failed to create MedicalConsultation instance: {str(e)}")
        raise
    
    def log_user_interaction(image, symptoms):
        """Log user interaction and process request"""
        logger.info("User started new consultation interaction")
        return consultation.process_consultation(image, symptoms)
    
    with gr.Blocks(
        title="Intelligent Medical Consultation System",
        theme=gr.themes.Soft(),
        css=".gradio-container {font-family: 'Segoe UI', Arial, sans-serif;}"
    ) as app:
        
        # Page title and description
        gr.Markdown("""
        # 🏥 Intelligent Medical Consultation System
        ## Multimodal Medical Diagnosis Assistant based on ERNIE 4.5
        
        > ⚠️ **Disclaimer**: This system is for reference only and cannot replace professional medical diagnosis. Seek immediate medical attention for emergencies.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Input Information")
                
                image_input = gr.Image(
                    label="Upload lesion image (optional)", 
                    type="filepath",
                    sources=["upload", "webcam"]
                )
                
                symptoms_input = gr.Textbox(
                    label="Please describe your symptoms in detail",
                    placeholder="E.g.: Fever 38.5°C, dry cough, fatigue for the past three days, mild sore throat...",
                    lines=6,
                    max_lines=10
                )
                
                with gr.Row():
                    submit_btn = gr.Button(
                        "🔍 Start Consultation", 
                        variant="primary", 
                        size="lg"
                    )
                    clear_btn = gr.Button(
                        "🗑️ Clear", 
                        variant="secondary"
                    )
            
            with gr.Column(scale=1):
                gr.Markdown("### 📋 Diagnosis Results")
                
                output = gr.Textbox(
                    label="AI Diagnostic Recommendations",
                    lines=25,
                    max_lines=30,
                    show_copy_button=True,
                    interactive=False
                )
        
        # Usage examples
        gr.Markdown("""
        ### 💡 Usage Tips
        
        1. **Text Description**: Please describe symptoms in detail, including duration, severity, accompanying symptoms, etc.
        2. **Image Upload**: You may upload relevant medical images (skin lesions, test reports, etc.)
        3. **Comprehensive Analysis**: The system will perform multimodal analysis combining text and images
        4. **Professional Advice**: Receive symptom analysis, risk assessment and preliminary medical recommendations
        
        **Note**: This system cannot replace professional medical diagnosis. Please seek medical attention promptly based on recommendations.
        """)
        
        # Bind event handlers
        def on_submit_click(image, symptoms):
            interaction_time = datetime.now()
            session_id = interaction_time.strftime('%H%M%S_%f')[:9]
            
            logger.info(f"[UI-{session_id}] User clicked submit button")
            logger.debug(f"[UI-{session_id}] Submission time: {interaction_time.isoformat()}")
            logger.debug(f"[UI-{session_id}] Submission parameters:")
            logger.debug(f"[UI-{session_id}] - Image: {'Yes' if image else 'No'}")
            logger.debug(f"[UI-{session_id}] - Symptom description length: {len(symptoms) if symptoms else 0}")
            
            try:
                result = log_user_interaction(image, symptoms)
                processing_time = (datetime.now() - interaction_time).total_seconds()
                logger.debug(f"[UI-{session_id}] Interface processing completed, time taken: {processing_time:.3f} seconds")
                return result
            except Exception as e:
                error_time = (datetime.now() - interaction_time).total_seconds()
                logger.error(f"[UI-{session_id}] Interface processing failed: {str(e)}")
                logger.debug(f"[UI-{session_id}] Error occurred after: {error_time:.3f} seconds")
                logger.debug(f"[UI-{session_id}] Error details: {repr(e)}", exc_info=True)
                return f"Interface processing error: {str(e)}"
        
        def on_clear_click():
            clear_time = datetime.now()
            clear_id = clear_time.strftime('%H%M%S_%f')[:9]
            
            logger.info(f"[UI-{clear_id}] User clicked clear button")
            logger.debug(f"[UI-{clear_id}] Clear operation time: {clear_time.isoformat()}")
            logger.debug(f"[UI-{clear_id}] Executing interface clear operation")
            
            return None, "", ""
        
        submit_btn.click(
            fn=on_submit_click,
            inputs=[image_input, symptoms_input],
            outputs=output,
            show_progress=True
        )
        
        clear_btn.click(
            fn=on_clear_click,
            outputs=[image_input, symptoms_input, output]
        )
        
        # Add keyboard shortcut support
        symptoms_input.submit(
            fn=on_submit_click,
            inputs=[image_input, symptoms_input],
            outputs=output,
            show_progress=True
        )
    
    logger.info("Gradio user interface created successfully")
    return app

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("Intelligent Medical Consultation System starting")
    logger.info("="*60)
    
    try:
        app = create_ui()
        
        # Launch configuration
        launch_config = {
            "share": False,
            "show_error": True,
            "quiet": False
        }
        
        logger.info(f"Gradio application launch configuration: {launch_config}")
        logger.info("Application starting, please wait...")
        
        app.launch(**launch_config)
        
    except KeyboardInterrupt:
        logger.info("User manually stopped the application")
    except Exception as e:
        logger.error(f"Application startup failed: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("Intelligent Medical Consultation System has shut down")