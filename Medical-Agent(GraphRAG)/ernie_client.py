import os
import base64
import json
import re
from openai import OpenAI
from typing import List, Dict, Optional, Tuple
from graph_kb import GraphKnowledgeBase
from graph_schema import ENTITY_TYPES, RELATION_TYPES


class ErnieClient:
    def __init__(self, 
                 host: str = "0.0.0.0", 
                 port: str = "8180", 
                 model_name: str = "local-vl-model",
                 graph_kb: Optional[GraphKnowledgeBase] = None):
        """初始化本地化VL模型客户端，支持多模态输入"""
        self.host = host
        self.port = port
        self.model_name = model_name
        self.base_url = f"http://{host}:{port}/v1"
        self.graph_kb = graph_kb
        
        self.client = OpenAI(
            api_key="null",
            base_url=self.base_url
        )

    def encode_image(self, image_path: str) -> str:
        """图片转Base64编码"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            raise Exception(f"图像编码失败: {str(e)}")

    def chat_completion(self, messages: List[Dict], stream: bool = True) -> str:
        """基础对话接口"""
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=stream,
                temperature=0.3
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
            return f"模型请求失败: {str(e)}"

    def text_generation(self, prompt: str, system_prompt: str = None) -> str:
        """纯文本生成"""
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
        
        return self.chat_completion(messages, stream=False)

    def analyze_image_and_text(self, text: str = None, image_path: str = None) -> str:
        """多模态分析（文本+图像）"""
        if not text and not image_path:
            return "请提供文本或图像进行分析"
        
        content = []
        
        if image_path:
            try:
                base64_image = self.encode_image(image_path)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    }
                })
            except Exception as e:
                return f"图像处理失败: {str(e)}"
        
        if text:
            content.append({
                "type": "text",
                "text": text
            })
        
        messages = [{
            "role": "user",
            "content": content
        }]
        
        return self.chat_completion(messages, stream=False)

    def analyze_symptoms(self, text: str) -> List[str]:
        """从自然语言文本中提取症状（优化版：严格限制只提取明确提到的症状）"""
        prompt = f"""从以下医疗描述中提取关键症状，严格遵循：
1. 仅提取文本中**明确提到**的症状，**绝对不添加任何未提及的症状**
2. 使用标准医学术语（如用"发热"而非"身体热"，"咳嗽"而非"嗓子痒"）
3. 仅返回Python字符串列表，无额外文本（例如：["发热", "干咳", "乏力"]）
4. 最多提取8个最关键的症状（忽略无关描述）
5. 排除疾病名称（如"流感"是疾病，"高热"是症状）
6. 如果未提到任何症状，返回空列表[]

患者描述：{text}

直接返回症状列表（无需代码块）："""
        
        result = self.text_generation(prompt)
        
        try:
            result = result.strip().replace("'", '"').replace("\n", "")
            match = re.search(r'\[(.*?)\]', result)
            if match:
                symptoms = json.loads(f"[{match.group(1)}]")
                # 额外过滤：只保留文本中明确出现的症状
                filtered = []
                for s in symptoms:
                    # 处理同义词匹配（如"小红点"对应"皮疹"）
                    synonyms = {
                        "小红点": "皮疹",
                        "痒": "瘙痒",
                        "疼": "疼痛",
                        "红点": "皮疹"
                    }
                    normalized = synonyms.get(s, s)
                    # 检查原始文本是否包含该症状或其同义词
                    if any(keyword in text for keyword in [s, normalized] + list(synonyms.keys())):
                        filtered.append(normalized)
                return list(set(filtered))  # 去重
        except Exception as e:
            print(f"症状解析错误: {e}")
            return []
        
        return []

    def extract_symptoms_from_multimodal(self, text: Optional[str] = None, image_path: Optional[str] = None) -> List[str]:
        """从自然语言和图像中联合提取症状（优化版：增加交叉验证）"""
        symptoms = []
        
        # 1. 从文本提取症状（作为基准）
        text_symptoms = []
        if text and text.strip():
            text_symptoms = self.analyze_symptoms(text)
            symptoms.extend(text_symptoms)
            print(f"从文本提取症状：{text_symptoms}")
        
        # 2. 从图像提取症状（增加严格约束）
        image_symptoms = []
        if image_path:
            try:
                # 图像分析提示词优化：只描述可见特征，不推测未显示症状
                image_analysis = self.medical_image_analysis(
                    image_path,
                    custom_prompt="""作为专业医疗影像分析师，请仅描述图像中**明确可见**的皮肤症状特征：
1. 病灶形态（如：红色丘疹、斑疹、水疱）
2. 颜色特征（如：红色、淡粉色）
3. 分布情况（如：散在分布、局部聚集）
4. 绝对不要添加任何图像中未显示的症状或进行疾病推测
5. 用客观描述，避免主观判断"""
                )
                print(f"图像分析结果：{image_analysis[:100]}...")
                
                # 从图像分析中提取症状（严格模式）
                prompt = f"""从以下图像分析结果中提取皮肤症状关键词，严格遵循：
1. 仅提取**明确描述**的症状，不添加任何推测性症状
2. 每个症状必须是图像中可见的（如"皮疹"、"红斑"）
3. 仅返回Python列表，无额外文本
4. 如果没有明确症状，返回空列表[]

图像分析结果：{image_analysis}

直接返回症状列表："""
                image_symptoms = self.text_generation(prompt)
                
                # 解析并过滤图像症状
                image_symptoms = image_symptoms.strip().replace("'", '"')
                if image_symptoms.startswith("```"):
                    image_symptoms = image_symptoms.split("```")[1].strip()
                image_symptoms = json.loads(image_symptoms)
                image_symptoms = [s for s in image_symptoms if isinstance(s, str) and s]
                
                print(f"从图像提取症状：{image_symptoms}")
            except Exception as e:
                print(f"图像症状提取失败：{e}")
        
        # 3. 交叉验证：如果文本中已有相关症状，才添加图像症状（减少误判）
        # 例如：文本提到"小红点"，图像提取"皮疹"则保留；图像单独提取"发热"则排除
        validated_symptoms = []
        for sym in image_symptoms:
            # 检查是否与文本症状相关
            is_related = any(
                sym in text_symptom or text_symptom in sym 
                for text_symptom in text_symptoms
            )
            # 对于完全无文本输入的情况，直接保留图像症状
            if is_related or not text_symptoms:
                validated_symptoms.append(sym)
        
        # 合并并去重
        all_symptoms = list(set(symptoms + validated_symptoms))
        
        # 最终过滤：移除明显不相关的症状（针对皮肤症状的特殊处理）
        skin_related = {"皮疹", "红斑", "丘疹", "瘙痒", "疼痛", "红点", "斑疹", "水疱"}
        filtered = [s for s in all_symptoms if s in skin_related]
        
        return filtered if filtered else all_symptoms

    def extract_relations(self, medical_text: str) -> List[Dict]:
        """从文本中抽取实体关系（支持新的实体和关系类型）"""
        if not medical_text:
            return []
        
        entity_types_str = "\n".join([f"- {k}: {v}" for k, v in ENTITY_TYPES.items()])
        relation_types_str = "\n".join([f"- {k}: {v}" for k, v in RELATION_TYPES.items()])
        
        prompt = f"""作为医疗知识工程师，从以下文本中提取实体-关系对。
严格遵循规则：

1. 实体类型（仅使用这些，不自定义）：
{entity_types_str}

2. 关系类型（仅使用这些，不自定义）：
{relation_types_str}

3. 输出格式：
返回Python字典列表，每个字典包含键：
"source"（实体名）、"source_type"（实体类型）、
"target"（实体名）、"target_type"（实体类型）、
"relation"（关系类型）。

医疗文本：{medical_text}

直接返回关系列表（无额外文本）："""
        
        result = self.text_generation(prompt)
        
        try:
            result = result.strip()
            if not result.startswith("["):
                result = re.sub(r'^.*?\[', '[', result)
            if not result.endswith("]"):
                result = re.sub(r'\].*$', ']', result)
            
            relations = json.loads(result.replace("'", '"'))
            valid_relations = []
            for rel in relations:
                if all(k in rel for k in ["source", "source_type", "target", "target_type", "relation"]) and \
                   rel["source_type"] in ENTITY_TYPES and \
                   rel["target_type"] in ENTITY_TYPES and \
                   rel["relation"] in RELATION_TYPES:
                    valid_relations.append(rel)
            return valid_relations
        except Exception as e:
            print(f"关系抽取失败：{e}")
            return []

    def enhance_with_graph_context(self, entity_list: List[str], entity_type: str) -> str:
        """从知识图谱获取上下文增强"""
        if not self.graph_kb or not entity_list:
            return "无可用图谱知识"
        
        graph_context = []
        for entity in entity_list:
            related_relations = self.graph_kb.query_related_entities(entity)
            if not related_relations:
                continue
            
            for src, rel, tgt, tgt_type in related_relations:
                relation_str = f"{src}（{self.graph_kb.graph.nodes[src]['type']}）→{rel}→ {tgt}（{tgt_type}）"
                graph_context.append(relation_str)
        
        unique_context = list(set(graph_context))
        return "\n".join(unique_context) if unique_context else "图谱中无相关知识"

    def analyze_risk(self, symptoms: List[str], medical_info: Dict = None) -> Dict:
        """风险分析（集成图谱上下文）"""
        if not symptoms:
            return {
                "risk_level": 1,
                "urgency": "请详细描述症状",
                "recommendations": ["请提供详细症状以进行准确评估"]
            }
        
        graph_context = ""
        if medical_info and "graph_kb" in medical_info:
            self.graph_kb = medical_info["graph_kb"]
            graph_context = self.enhance_with_graph_context(symptoms, entity_type="Symptom")
        
        prompt = f"""作为专业医疗风险评估师，基于症状和图谱知识评估风险。

【症状列表】：{', '.join(symptoms)}

【图谱知识上下文】：
{graph_context}

返回Python字典，包含：
- risk_level: 1-5（1=极低，2=低，3=中，4=高，5=紧急）
- urgency: 就医建议（如"紧急就诊"、"常规门诊"）
- recommendations: 3-5条具体建议（避免模糊表述）

示例：
{{
    "risk_level": 4,
    "urgency": "建议24小时内紧急就诊",
    "recommendations": [
        "前往呼吸科进行流感病毒检测",
        "避免与家人密切接触以防传播",
        "体温超过38.5℃时服用退烧药",
        "每日饮水1.5-2L预防脱水"
    ]
}}

直接返回字典（无额外文本）："""
        
        result = self.text_generation(prompt)
        return self._parse_medical_dict(result)

    def generate_treatment_plan(self, symptoms: List[str], medical_info: Dict = None) -> Dict:
        """生成治疗方案（集成图谱上下文）"""
        if not symptoms:
            return {
                "examinations": ["请先提供详细症状"],
                "medications": ["药物需根据具体诊断确定"],
                "lifestyle": ["保持健康生活方式"]
            }
        
        graph_context = ""
        if medical_info and "graph_kb" in medical_info:
            self.graph_kb = medical_info["graph_kb"]
            graph_context = self.enhance_with_graph_context(symptoms, entity_type="Symptom")
        
        prompt = f"""作为医疗顾问，基于症状和图谱知识生成治疗方案。

【症状列表】：{', '.join(symptoms)}

【图谱知识上下文】（疾病-治疗-检查关系）：
{graph_context}

返回Python字典，包含：
- examinations: 3-4项推荐检查（匹配图谱知识）
- medications: 3条安全建议（不含处方药，需注明"遵医嘱"）
- lifestyle: 4-5条可操作的生活建议（针对症状定制）

直接返回字典（无额外文本）："""
        
        result = self.text_generation(prompt)
        return self._parse_medical_dict(result)

    def _parse_medical_dict(self, result: str) -> Dict:
        """解析医疗相关字典输出"""
        try:
            result = result.strip()
            if result.startswith("```"):
                result = result.split("```")[1].strip()
            if result.startswith(("python", "json")):
                result = result.split("\n", 1)[1].strip()
            
            try:
                return json.loads(result)
            except:
                return eval(result)
        except Exception as e:
            print(f"字典解析失败：{e}")
            return {
                "risk_level": 2,
                "urgency": "建议就医咨询",
                "examinations": ["相关专科检查", "基础体检项目"],
                "medications": ["遵医嘱用药", "避免自行用药"],
                "lifestyle": ["充分休息", "均衡饮食", "监测症状"]
            }

    def test_connection(self) -> bool:
        """测试模型连接"""
        try:
            response = self.text_generation("返回'OK'确认连接")
            return "OK" in response.strip()
        except:
            return False

    def extract_entities_from_text(self, text: str) -> List[str]:
        """从医疗文本中提取关键实体（疾病、症状、治疗方法等）"""
        if not text:
            return []
        
        prompt = f"""从以下医疗诊断报告中提取关键实体，包括但不限于：
        - 疾病名称
        - 症状
        - 治疗方法
        - 检查项目
        - 身体部位
        
        仅返回实体列表，每个实体一行，不要添加任何解释或说明：

{text}

实体列表："""
        
        result = self.text_generation(prompt)
        # 解析结果，过滤空行和无效内容
        entities = [line.strip() for line in result.split('\n') if line.strip() and not line.strip().startswith('-')]
        return list(set(entities))  # 去重

    def medical_image_analysis(self, image_path: str, custom_prompt: str = None) -> str:
        """医疗图像分析"""
        if not image_path or not os.path.exists(image_path):
            return "图像路径不存在"
        
        base64_image = self.encode_image(image_path)
        
        prompt = custom_prompt if custom_prompt else """作为专业医疗影像分析师，请分析此图像并提供详细的医学解读，
包括可见的异常特征、可能的诊断方向和建议的进一步检查。请保持客观，不做确定性诊断。"""
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ]
        
        return self.chat_completion(messages, stream=False)
