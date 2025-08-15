from knowledge_base import KnowledgeBase

def initialize_medical_knowledge():
    kb = KnowledgeBase()
    
    # Add symptom knowledge
    symptoms_docs = [
        "Fever is a defensive response of the body to infection or other stimuli; a body temperature exceeding 37.3℃ is considered fever.",
        "Cough is an important defensive reflex action, which can be divided into acute cough and chronic cough.",
        "Fatigue is a subjective feeling of tiredness that may be associated with various diseases."
    ]
    symptoms_meta = [
        {"type": "symptom", "severity": "medium"},
        {"type": "symptom", "severity": "medium"},
        {"type": "symptom", "severity": "low"}
    ]
    symptoms_ids = ["symptom_1", "symptom_2", "symptom_3"]
    
    # Add disease knowledge
    diseases_docs = [
        "Common cold: A common upper respiratory tract infection, main symptoms include nasal congestion, runny nose, sore throat, etc.",
        "Influenza: Caused by influenza viruses, symptoms are more severe than the common cold, often accompanied by high fever and generalized body aches.",
        "Novel coronavirus infection: Highly contagious, can cause symptoms such as fever, dry cough, and fatigue."
    ]
    diseases_meta = [
        {"type": "disease", "risk_level": 1},
        {"type": "disease", "risk_level": 2},
        {"type": "disease", "risk_level": 3}
    ]
    diseases_ids = ["disease_1", "disease_2", "disease_3"]
    
    # Add treatment knowledge
    treatments_docs = [
        "Common cold treatment: Rest, adequate fluid intake, and use of cold medications if necessary.",
        "Influenza treatment: Antiviral drugs (such as oseltamivir) and symptomatic treatment.",
        "COVID-19 treatment: Isolation treatment, with corresponding measures taken according to the severity of symptoms."
    ]
    treatments_meta = [
        {"type": "treatment", "urgency": "low"},
        {"type": "treatment", "urgency": "medium"},
        {"type": "treatment", "urgency": "high"}
    ]
    treatments_ids = ["treatment_1", "treatment_2", "treatment_3"]
    
    # Add to knowledge base
    kb.add_medical_knowledge("symptoms", symptoms_docs, symptoms_meta, symptoms_ids)
    kb.add_medical_knowledge("diseases", diseases_docs, diseases_meta, diseases_ids)
    kb.add_medical_knowledge("treatments", treatments_docs, treatments_meta, treatments_ids)

if __name__ == "__main__":
    initialize_medical_knowledge()
    print("Medical knowledge base initialization completed!")