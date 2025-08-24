# graph_schema.py
"""定义医疗知识图谱中的实体类型和关系类型"""

ENTITY_TYPES = {
    "Symptom": "症状（如发热、咳嗽、头痛等）",
    "Disease": "疾病（如感冒、流感、肺炎等）",
    "Treatment": "治疗方法（如药物、手术、理疗等）",
    "Examination": "检查项目（如血常规、CT、X光等）",
    "BodyPart": "身体部位（如肺部、心脏、喉咙等）",
    "Medication": "药物（如阿司匹林、布洛芬等）",
    "RiskFactor": "风险因素（如吸烟、肥胖等）"
}

RELATION_TYPES = {
    "CAUSES": "导致（疾病→症状，风险因素→疾病）",
    "TREATS": "治疗（治疗方法→疾病，药物→疾病/症状）",
    "REQUIRES": "需要（疾病→检查项目）",
    "AFFECTS": "影响（疾病→身体部位）",
    "ACCOMPANIES": "伴随（症状→症状）",
    "DIAGNOSES": "诊断（检查项目→疾病）",
    "PREVENTS": "预防（措施→疾病）",
    "HAS_SYMPTOM": "有症状（疾病→症状）"
}

# 实体类型颜色映射，用于可视化
ENTITY_COLORS = {
    "Symptom": "#FF9999",
    "Disease": "#66B2FF",
    "Treatment": "#99FF99",
    "Examination": "#FFCC99",
    "BodyPart": "#FF99CC",
    "Medication": "#CC99FF",
    "RiskFactor": "#FFFF99"
}
