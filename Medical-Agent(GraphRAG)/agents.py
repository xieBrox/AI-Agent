import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from graph_kb import GraphKnowledgeBase
from ernie_client import ErnieClient

# 配置日志
logging.basicConfig(level=logging.INFO)
agents_logger = logging.getLogger("medical_agents")

class KnowledgeRetrievalAgent:
    """知识检索代理，从知识图谱中检索相关信息"""
    
    def __init__(self, graph_kb: GraphKnowledgeBase):
        agents_logger.debug("初始化基于图的知识检索代理")
        self.graph_kb = graph_kb
        agents_logger.info("知识检索代理初始化完成")
    
    def retrieve_relevant_info(self, symptoms: List[str], max_hops: int = 2) -> Dict:
        """从知识图谱检索与症状相关的信息"""
        retrieve_start_time = datetime.now()
        retriever_id = retrieve_start_time.strftime('%H%M%S_%f')[:9]
        agents_logger.info(f"[Retriever-{retriever_id}] 开始检索症状相关知识：{symptoms}")
        
        # 1. 从症状找到可能的疾病
        possible_diseases = set()
        symptom_relations = {}
        
        for symptom in symptoms:
            relations = self.graph_kb.query_related_entities(symptom, max_hops=1)
            symptom_relations[symptom] = relations
            
            for src, rel, tgt, tgt_type in relations:
                if tgt_type == "Disease" and rel in ["HAS_SYMPTOM", "CAUSES"]:
                    possible_diseases.add(src)
                elif src == symptom and tgt_type == "Disease" and rel in ["CAUSES"]:
                    possible_diseases.add(tgt)
        
        # 2. 获取每个疾病的详细信息
        disease_info = {}
        for disease in possible_diseases:
            # 症状
            disease_symptoms = [
                tgt for src, rel, tgt, tgt_type in 
                self.graph_kb.query_related_entities(disease, relation="HAS_SYMPTOM")
                if tgt_type == "Symptom"
            ]
            
            # 治疗方法
            treatments = [
                src for src, rel, tgt, tgt_type in 
                self.graph_kb.query_related_entities(disease, relation="TREATS")
                if tgt_type == "Disease" and src in ["Treatment", "Medication"]
            ]
            
            # 检查项目
            examinations = [
                tgt for src, rel, tgt, tgt_type in 
                self.graph_kb.query_related_entities(disease, relation="REQUIRES")
                if tgt_type == "Examination"
            ]
            
            # 影响的身体部位
            affected_body_parts = [
                tgt for src, rel, tgt, tgt_type in 
                self.graph_kb.query_related_entities(disease, relation="AFFECTS")
                if tgt_type == "BodyPart"
            ]
            
            disease_info[disease] = {
                "symptoms": disease_symptoms,
                "treatments": treatments,
                "examinations": examinations,
                "affected_body_parts": affected_body_parts
            }
        
        # 3. 查找症状之间的关联
        symptom_connections = []
        for i, symptom1 in enumerate(symptoms):
            for symptom2 in symptoms[i+1:]:
                paths = self.graph_kb.find_paths(symptom1, symptom2, max_length=2)
                if paths:
                    symptom_connections.append({
                        "symptom1": symptom1,
                        "symptom2": symptom2,
                        "paths": paths
                    })
        
        # 4. 构建结果
        result = {
            "symptoms": symptoms,
            "possible_diseases": list(possible_diseases),
            "disease_info": disease_info,
            "symptom_relations": symptom_relations,
            "symptom_connections": symptom_connections,
            "retrieval_time": (datetime.now() - retrieve_start_time).total_seconds()
        }
        
        agents_logger.info(f"[Retriever-{retriever_id}] 检索完成，耗时 {result['retrieval_time']:.2f}s")
        return result


class DiagnosisAgent:
    """诊断代理，基于检索到的知识进行诊断推理"""
    
    def __init__(self, ernie_client: ErnieClient, retrieval_agent: KnowledgeRetrievalAgent):
        agents_logger.debug("初始化诊断代理")
        self.ernie = ernie_client
        self.retrieval_agent = retrieval_agent
        agents_logger.info("诊断代理初始化完成")
    
    def process_symptoms(self, symptoms: List[str], medical_history: str = "") -> Dict:
        """处理症状并生成诊断建议"""
        diag_start_time = datetime.now()
        diag_id = diag_start_time.strftime('%H%M%S_%f')[:9]
        agents_logger.info(f"[Diagnosis-{diag_id}] 开始诊断，症状：{symptoms}")
        
        # 1. 检索相关知识
        graph_context = self.retrieval_agent.retrieve_relevant_info(symptoms)
        
        # 2. 分析风险等级（传入知识图谱增强推理）
        risk_analysis = self.analyze_risk_level(symptoms, graph_context)
        
        # 3. 生成诊断建议
        diagnosis = self.generate_diagnosis(symptoms, medical_history, graph_context)
        
        # 4. 构建完整结果
        result = {
            "symptoms": symptoms,
            "medical_history": medical_history,
            "risk_analysis": risk_analysis,
            "diagnosis": diagnosis,
            "graph_context": graph_context,
            "diagnosis_time": (datetime.now() - diag_start_time).total_seconds(),
            "full_report": self._generate_full_report(symptoms, medical_history, risk_analysis, diagnosis)
        }
        
        agents_logger.info(f"[Diagnosis-{diag_id}] 诊断完成，耗时 {result['diagnosis_time']:.2f}s")
        return result
    
    def analyze_risk_level(self, symptoms: List[str], medical_info: Dict) -> Dict:
        """分析症状的风险等级"""
        # 提取图中的关键关系链
        relation_chains = []
        for disease in medical_info.get("possible_diseases", []):
            for symptom in symptoms:
                paths = self.retrieval_agent.graph_kb.find_paths(disease, symptom, max_length=2)
                for path in paths:
                    chain = " → ".join([f"{p[0]}[{p[1]}]" for p in path])
                    relation_chains.append(f"{chain} → {symptom}")
        
        # 构建风险分析的上下文
        graph_context = {
            "relation_chains": relation_chains,
            "possible_diseases_count": len(medical_info.get("possible_diseases", [])),
            "high_risk_indicators": self._identify_high_risk_indicators(symptoms, medical_info)
        }
        
        # 调用ERNIE进行风险分析（传入知识图谱）
        return self.ernie.analyze_risk(
            symptoms=symptoms,
            medical_info={"graph_kb": self.retrieval_agent.graph_kb, **graph_context}
        )
    
    def generate_diagnosis(self, symptoms: List[str], medical_history: str, graph_context: Dict) -> Dict:
        """生成详细诊断方案"""
        # 调用ERNIE生成治疗方案（传入知识图谱）
        return self.ernie.generate_treatment_plan(
            symptoms=symptoms,
            medical_info={"graph_kb": self.retrieval_agent.graph_kb,** graph_context}
        )
    
    def _identify_high_risk_indicators(self, symptoms: List[str], medical_info: Dict) -> List[str]:
        """识别高风险指标"""
        high_risk_indicators = []
        
        # 高风险症状列表
        critical_symptoms = ["高热", "呼吸困难", "剧烈头痛", "胸痛", "意识模糊"]
        for symptom in symptoms:
            if symptom in critical_symptoms:
                high_risk_indicators.append(f"出现高风险症状: {symptom}")
        
        # 严重疾病匹配
        severe_diseases = ["肺炎", "心肌梗死", "中风", "脑膜炎"]
        severe_matches = [d for d in medical_info.get("possible_diseases", []) if d in severe_diseases]
        if len(severe_matches) > 0:
            high_risk_indicators.append(f"可能存在严重疾病: {', '.join(severe_matches)}")
        
        # 症状组合风险
        if "高热" in symptoms and "呼吸困难" in symptoms:
            high_risk_indicators.append("高热伴随呼吸困难，可能提示严重感染")
        
        return high_risk_indicators
    
    def _generate_full_report(self, symptoms: List[str], medical_history: str, risk_analysis: Dict, diagnosis: Dict) -> str:
        """生成完整的诊断报告文本"""
        report = []
        report.append("## 诊断报告")
        report.append(f"### 症状: {', '.join(symptoms)}")
        if medical_history:
            report.append(f"### 病史: {medical_history}")
        report.append(f"### 风险等级: {risk_analysis['risk_level']} ({risk_analysis['urgency']})")
        report.append("### 推荐检查:")
        for exam in diagnosis['examinations']:
            report.append(f"- {exam}")
        report.append("### 建议用药:")
        for med in diagnosis['medications']:
            report.append(f"- {med}")
        report.append("### 生活建议:")
        for life in diagnosis['lifestyle']:
            report.append(f"- {life}")
        return "\n".join(report)


class VisualizationAgent:
    """可视化代理，根据诊断报告生成知识图谱可视化"""
    
    def __init__(self, graph_kb: GraphKnowledgeBase, ernie_client: ErnieClient):
        agents_logger.debug("初始化可视化代理")
        self.graph_kb = graph_kb
        self.ernie = ernie_client
        agents_logger.info("可视化代理初始化完成")
    
    def generate_visualization_from_report(self, diagnosis_report: str, filename: str = "diagnosis_related_kg.html") -> str:
        """从诊断报告中提取实体并生成相关知识图谱可视化"""
        if not self.graph_kb:
            return "知识图谱尚未初始化，无法生成可视化"
        
        if not diagnosis_report:
            return "诊断报告为空，无法生成可视化"
        
        # 从诊断报告中提取关键实体（疾病、症状、治疗方法等）
        entities = self.ernie.extract_entities_from_text(diagnosis_report)
        
        if not entities:
            return "未能从诊断报告中提取到实体，无法生成知识图谱"
        
        # 生成可视化文件，高亮报告中提到的实体
        self.graph_kb.visualize(
            filename=filename,
            highlight_entities=entities,
            max_nodes=100
        )
        
        return f"已生成与诊断报告相关的知识图谱：<a href='{filename}' target='_blank'>查看图谱</a>"
