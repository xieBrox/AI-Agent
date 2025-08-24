"""初始化医疗知识图谱，从文本和结构化数据中抽取实体关系（含皮疹相关内容）"""
import os
import json
import pickle
from typing import List
from graph_kb import GraphKnowledgeBase
from ernie_client import ErnieClient

def load_medical_texts(data_dir: str = "medical_data") -> List[str]:
    """加载医疗文本数据（新增皮疹相关疾病文本）"""
    texts = []
    
    # 创建示例数据（如果目录不存在，新增皮疹相关疾病）
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
        example_data = {
            "普通感冒": "普通感冒是常见上呼吸道感染，症状有鼻塞、流涕、咳嗽、喉咙痛。由病毒引起，治疗包括休息、多喝水和对症治疗如解热镇痛药。影响鼻腔和喉咙。",
            "流感": "流感是流感病毒引起的急性呼吸道传染病，症状包括高热、头痛、乏力、肌肉酸痛。严重时引发肺炎。治疗用奥司他韦，预防措施包括接种疫苗。",
            "肺炎": "肺炎是肺部炎症，由细菌、病毒或真菌引起。症状有发热、咳嗽、呼吸困难、胸痛。诊断需要胸部X光和血常规。严重者需住院和抗生素治疗。",
            "高血压": "高血压是动脉血压持续升高的慢性病，通常无症状，但长期不控制会增加心脏病和中风风险。治疗包括ACE抑制剂等药物和低盐饮食、规律运动。",
            # 新增：皮疹相关疾病文本（含定义、症状、病因、治疗）
            "湿疹": "湿疹是慢性炎症性皮肤病，核心症状为皮肤红斑、瘙痒、丘疹，严重时出现渗液、结痂。病因与过敏体质、皮肤屏障受损、环境刺激（如干燥、尘螨）相关。治疗包括外用氢化可的松乳膏（轻度）、口服抗组胺药（止痒），需避免搔抓和接触过敏原。主要影响四肢屈侧、面部等皮肤区域。",
            "荨麻疹": "荨麻疹（风团）是皮肤黏膜暂时性水肿反应，症状为突发风团、剧烈瘙痒，风团可在数小时内消退且不留痕迹。多由食物过敏（如海鲜、芒果）、药物过敏（如青霉素）或感染诱发。治疗首选口服氯雷他定、西替利嗪等抗组胺药，严重时需用糖皮质激素。影响全身皮肤，可累及眼睑、口唇（血管性水肿）。",
            "接触性皮炎": "接触性皮炎是皮肤接触外界物质后引发的炎症，分为刺激性（如强酸、洗涤剂）和过敏性（如金属镍、化妆品）两类。症状为接触部位红斑、水疱、瘙痒、灼热感，边界与接触物一致。治疗需立即脱离接触物，外用炉甘石洗剂（止痒）或糠酸莫米松乳膏，严重时口服泼尼松。主要影响手部、面部等暴露接触部位皮肤。",
            "药疹": "药疹是药物通过口服、注射等途径进入人体后引发的皮肤黏膜反应，症状多样，包括红斑、丘疹、水疱（严重型如史蒂文斯-约翰逊综合征），常伴瘙痒或发热。常见致病药物有抗生素（青霉素）、解热镇痛药（对乙酰氨基酚）、抗癫痫药（卡马西平）。治疗需立即停用致敏药物，轻症用抗组胺药，重症需住院用大剂量糖皮质激素。可影响全身皮肤，严重时累及黏膜（口腔、眼结膜）。",
            "水痘": "水痘是水痘-带状疱疹病毒引起的传染病，儿童多见，症状除发热、乏力外，特征性表现为全身分批出现的皮疹（斑疹→丘疹→水疱→结痂），皮疹伴轻微瘙痒。治疗以对症为主，如炉甘石洗剂止痒、对乙酰氨基酚退热，预防需接种水痘疫苗。皮疹主要分布于躯干、头面部皮肤，可累及口腔黏膜。"
        }
        
        with open(os.path.join(data_dir, "disease_info.json"), "w", encoding="utf-8") as f:
            json.dump(example_data, f, ensure_ascii=False, indent=2)
        
        print(f"已创建示例数据（含皮疹相关疾病）到 {data_dir} 目录")
    
    # 读取数据文件（兼容原有逻辑，自动加载新增的皮疹文本）
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
                texts.append(f.read())
        elif filename.endswith(".json"):
            with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    for value in data.values():
                        if isinstance(value, str):
                            texts.append(value)
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, str):
                            texts.append(item)
    
    return texts

def initialize_medical_knowledge(knowledge_file: str = "medical_kb.pkl", 
                                rebuild: bool = False,
                                model_host: str = "0.0.0.0",
                                model_port: str = "8180") -> GraphKnowledgeBase:
    """初始化医疗知识图谱（含皮疹相关实体关系）"""
    # 加载已有图谱（如果存在且不重建）
    if os.path.exists(knowledge_file) and not rebuild:
        graph_kb = GraphKnowledgeBase()
        if graph_kb.load_from_file(knowledge_file):
            return graph_kb
        print("加载现有知识图谱失败，将重新构建（含皮疹相关内容）")
    
    # 创建新的知识图谱
    graph_kb = GraphKnowledgeBase()
    
    # 初始化ERNIE客户端
    ernie = ErnieClient(
        host=model_host,
        port=model_port,
        model_name="local-vl-model",
        graph_kb=graph_kb
    )
    
    # 测试模型连接
    if not ernie.test_connection():
        print("警告：无法连接到本地模型服务，将仅使用手动定义的关系（含皮疹）")
    else:
        print("成功连接到本地模型服务，将抽取皮疹相关文本的实体关系")
    
    # 1. 加载医疗文本数据（含新增的皮疹相关疾病文本）
    medical_texts = load_medical_texts()
    print(f"加载了 {len(medical_texts)} 条医疗文本数据（含5条皮疹相关疾病文本）")
    
    # 2. 从文本中抽取实体关系（含皮疹相关关系）
    for i, text in enumerate(medical_texts):
        print(f"处理文本 {i+1}/{len(medical_texts)}...")
        
        try:
            relations = ernie.extract_relations(text)
            
            if not relations:
                print(f"文本 {i+1} 未抽取到关系")
                continue
            
            # 添加到知识图谱
            for rel in relations:
                try:
                    graph_kb.add_relation(
                        source=rel["source"],
                        source_type=rel["source_type"],
                        target=rel["target"],
                        target_type=rel["target_type"],
                        relation_type=rel["relation"]
                    )
                except Exception as e:
                    print(f"添加关系失败: {str(e)}, 关系: {rel}")
        except Exception as e:
            print(f"处理文本 {i+1} 时出错: {str(e)}")
    
    # 3. 添加手动定义的关系（重点补充皮疹相关实体关系）
    add_manual_relations(graph_kb)
    print(f"添加手动关系后（含皮疹）：实体数 {len(graph_kb.graph.nodes)}, 关系数 {len(graph_kb.graph.edges)}")
    
    # 4. 保存知识图谱
    graph_kb.save_to_file(knowledge_file)
    
    # 5. 生成可视化（可查看皮疹相关实体的关联）
    graph_kb.visualize("medical_knowledge_graph.html")
    
    return graph_kb

def add_manual_relations(graph_kb: GraphKnowledgeBase) -> None:
    """添加手动定义的关系（新增皮疹相关核心关系）"""
    # 原有关系保留
    # 症状-疾病
    graph_kb.add_relation("普通感冒", "Disease", "鼻塞", "Symptom", "HAS_SYMPTOM")
    graph_kb.add_relation("普通感冒", "Disease", "流涕", "Symptom", "HAS_SYMPTOM")
    graph_kb.add_relation("流感", "Disease", "高热", "Symptom", "HAS_SYMPTOM")
    graph_kb.add_relation("肺炎", "Disease", "呼吸困难", "Symptom", "HAS_SYMPTOM")
    
    # 疾病-治疗
    graph_kb.add_relation("休息", "Treatment", "普通感冒", "Disease", "TREATS")
    graph_kb.add_relation("奥司他韦", "Medication", "流感", "Disease", "TREATS")
    graph_kb.add_relation("抗生素", "Medication", "肺炎", "Disease", "TREATS")
    
    # 疾病-检查
    graph_kb.add_relation("肺炎", "Disease", "胸部X光", "Examination", "REQUIRES")
    graph_kb.add_relation("流感", "Disease", "病毒检测", "Examination", "REQUIRES")
    
    # 疾病-身体部位
    graph_kb.add_relation("普通感冒", "Disease", "鼻腔", "BodyPart", "AFFECTS")
    graph_kb.add_relation("肺炎", "Disease", "肺部", "BodyPart", "AFFECTS")

    graph_kb.add_relation("吸烟", "RiskFactor", "肺癌", "Disease", "CAUSES")
    graph_kb.add_relation("肥胖", "RiskFactor", "高血压", "Disease", "CAUSES")
    graph_kb.add_relation("运动", "Treatment", "肥胖", "RiskFactor", "TREATS")

    # --------------------------------------------------------------------------
    # 新增：皮疹相关核心关系（按“疾病-症状-治疗-检查-部位-风险因素”维度补充）
    # --------------------------------------------------------------------------
    # 1. 皮疹相关疾病 → 症状（HAS_SYMPTOM）
    graph_kb.add_relation("湿疹", "Disease", "红斑", "Symptom", "HAS_SYMPTOM")
    graph_kb.add_relation("湿疹", "Disease", "瘙痒", "Symptom", "HAS_SYMPTOM")
    graph_kb.add_relation("湿疹", "Disease", "丘疹", "Symptom", "HAS_SYMPTOM")
    graph_kb.add_relation("湿疹", "Disease", "渗液", "Symptom", "HAS_SYMPTOM")
    
    graph_kb.add_relation("荨麻疹", "Disease", "风团", "Symptom", "HAS_SYMPTOM")
    graph_kb.add_relation("荨麻疹", "Disease", "剧烈瘙痒", "Symptom", "HAS_SYMPTOM")
    graph_kb.add_relation("荨麻疹", "Disease", "血管性水肿", "Symptom", "HAS_SYMPTOM")
    
    graph_kb.add_relation("接触性皮炎", "Disease", "水疱", "Symptom", "HAS_SYMPTOM")
    graph_kb.add_relation("接触性皮炎", "Disease", "灼热感", "Symptom", "HAS_SYMPTOM")
    
    graph_kb.add_relation("药疹", "Disease", "水疱", "Symptom", "HAS_SYMPTOM")
    graph_kb.add_relation("药疹", "Disease", "发热", "Symptom", "HAS_SYMPTOM")
    graph_kb.add_relation("药疹", "Disease", "口腔黏膜损伤", "Symptom", "HAS_SYMPTOM")
    
    graph_kb.add_relation("水痘", "Disease", "皮疹（斑疹→丘疹→水疱→结痂）", "Symptom", "HAS_SYMPTOM")

    # 2. 治疗方式 → 皮疹相关疾病（TREATS）
    # 药物治疗
    graph_kb.add_relation("氢化可的松乳膏", "Medication", "湿疹", "Disease", "TREATS")
    graph_kb.add_relation("氯雷他定", "Medication", "荨麻疹", "Disease", "TREATS")
    graph_kb.add_relation("西替利嗪", "Medication", "荨麻疹", "Disease", "TREATS")
    graph_kb.add_relation("炉甘石洗剂", "Medication", "接触性皮炎", "Disease", "TREATS")
    graph_kb.add_relation("糠酸莫米松乳膏", "Medication", "接触性皮炎", "Disease", "TREATS")
    graph_kb.add_relation("泼尼松", "Medication", "接触性皮炎（严重）", "Disease", "TREATS")
    graph_kb.add_relation("糖皮质激素（大剂量）", "Medication", "药疹（重症）", "Disease", "TREATS")
    
    # 非药物治疗
    graph_kb.add_relation("避免接触过敏原", "Treatment", "湿疹", "Disease", "TREATS")
    graph_kb.add_relation("避免接触过敏原", "Treatment", "荨麻疹", "Disease", "TREATS")
    graph_kb.add_relation("停用致敏药物", "Treatment", "药疹", "Disease", "TREATS")

    # 3. 皮疹相关疾病 → 检查项目（REQUIRES）
    graph_kb.add_relation("荨麻疹", "Disease", "过敏原检测", "Examination", "REQUIRES")
    graph_kb.add_relation("接触性皮炎", "Disease", "过敏原检测", "Examination", "REQUIRES")
    graph_kb.add_relation("湿疹", "Disease", "皮肤镜检查", "Examination", "REQUIRES")
    graph_kb.add_relation("药疹", "Disease", "药物过敏原检测", "Examination", "REQUIRES")

    # 4. 皮疹相关疾病 → 影响身体部位（AFFECTS）
    graph_kb.add_relation("湿疹", "Disease", "四肢屈侧皮肤", "BodyPart", "AFFECTS")
    graph_kb.add_relation("湿疹", "Disease", "面部皮肤", "BodyPart", "AFFECTS")
    graph_kb.add_relation("荨麻疹", "Disease", "眼睑皮肤", "BodyPart", "AFFECTS")
    graph_kb.add_relation("荨麻疹", "Disease", "口唇黏膜", "BodyPart", "AFFECTS")
    graph_kb.add_relation("接触性皮炎", "Disease", "手部皮肤", "BodyPart", "AFFECTS")
    graph_kb.add_relation("接触性皮炎", "Disease", "面部皮肤", "BodyPart", "AFFECTS")
    graph_kb.add_relation("药疹", "Disease", "眼结膜", "BodyPart", "AFFECTS")
    graph_kb.add_relation("水痘", "Disease", "躯干皮肤", "BodyPart", "AFFECTS")
    graph_kb.add_relation("水痘", "Disease", "头面部皮肤", "BodyPart", "AFFECTS")

    # 5. 风险因素 → 皮疹相关疾病（CAUSES）
    graph_kb.add_relation("过敏体质", "RiskFactor", "湿疹", "Disease", "CAUSES")
    graph_kb.add_relation("皮肤屏障受损", "RiskFactor", "湿疹", "Disease", "CAUSES")
    graph_kb.add_relation("尘螨", "RiskFactor", "湿疹", "Disease", "CAUSES")
    
    graph_kb.add_relation("海鲜过敏", "RiskFactor", "荨麻疹", "Disease", "CAUSES")
    graph_kb.add_relation("芒果过敏", "RiskFactor", "荨麻疹", "Disease", "CAUSES")
    graph_kb.add_relation("青霉素过敏", "RiskFactor", "荨麻疹", "Disease", "CAUSES")
    
    graph_kb.add_relation("金属镍接触", "RiskFactor", "接触性皮炎", "Disease", "CAUSES")
    graph_kb.add_relation("化妆品刺激", "RiskFactor", "接触性皮炎", "Disease", "CAUSES")
    graph_kb.add_relation("强酸接触", "RiskFactor", "接触性皮炎", "Disease", "CAUSES")
    
    graph_kb.add_relation("青霉素", "RiskFactor", "药疹", "Disease", "CAUSES")
    graph_kb.add_relation("对乙酰氨基酚", "RiskFactor", "药疹", "Disease", "CAUSES")
    graph_kb.add_relation("卡马西平", "RiskFactor", "药疹", "Disease", "CAUSES")

    # 6. 皮疹（作为症状）→ 关联其他疾病（HAS_SYMPTOM，反向补充）
    graph_kb.add_relation("麻疹", "Disease", "皮疹", "Symptom", "HAS_SYMPTOM")
    graph_kb.add_relation("猩红热", "Disease", "皮疹", "Symptom", "HAS_SYMPTOM")
    graph_kb.add_relation("手足口病", "Disease", "皮疹", "Symptom", "HAS_SYMPTOM")

if __name__ == "__main__":
    initialize_medical_knowledge(
        rebuild=True,  # 重建图谱以加载新增的皮疹内容
        model_host="0.0.0.0",
        model_port="8180"
    )