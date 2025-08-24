import os
import logging
import gradio as gr
import json
from datetime import datetime
from typing import Dict, List
from init_knowledge_base import initialize_medical_knowledge
from agents import KnowledgeRetrievalAgent, DiagnosisAgent, VisualizationAgent
from ernie_client import ErnieClient

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medical_ai_system")

# 全局组件
graph_kb = None
retrieval_agent = None
diagnosis_agent = None
visualization_agent = None
ernie_client = None

def init_system():
    """初始化系统组件"""
    global graph_kb, retrieval_agent, diagnosis_agent, visualization_agent, ernie_client
    
    # 初始化知识图谱
    graph_kb = initialize_medical_knowledge()
    
    # 初始化ERNIE客户端（连接本地VL模型）
    ernie_client = ErnieClient(
        host="0.0.0.0",
        port="8180",
        model_name="local-vl-model",
        graph_kb=graph_kb
    )
    
    # 测试模型连接
    if not ernie_client.test_connection():
        logger.warning("无法连接到本地模型服务，功能可能受限")
        return "系统初始化完成，但模型连接失败，请检查本地服务"
    
    # 初始化智能代理
    retrieval_agent = KnowledgeRetrievalAgent(graph_kb)
    diagnosis_agent = DiagnosisAgent(ernie_client, retrieval_agent)
    visualization_agent = VisualizationAgent(graph_kb, ernie_client)  # 初始化可视化代理
    
    logger.info("系统初始化完成")
    return "系统初始化完成，可开始诊断"

def process_query(symptoms: List[str], medical_history: str) -> Dict:
    """处理提取到的症状列表"""
    if not symptoms:
        return {"error": "未提取到有效症状，请提供更详细的描述或图像"}
    
    # 调用诊断代理处理症状
    result = diagnosis_agent.process_symptoms(symptoms, medical_history)
    return result

def format_result(result: Dict) -> str:
    """格式化诊断结果为Markdown"""
    if "error" in result:
        return result["error"]
    
    report = []
    
    # 基本信息
    report.append("## 🏥 诊断报告")
    report.append(f"**提取的症状**: {', '.join(result['symptoms'])}")
    if result['medical_history']:
        report.append(f"**病史**: {result['medical_history']}")
    report.append("")
    
    # 图像分析结果（如果有）
    if "image_analysis" in result:
        report.append("### 📊 图像分析结果")
        report.append(result["image_analysis"])
        report.append("")
    
    # 风险分析
    report.append("### ⚠️ 风险评估")
    risk = result['risk_analysis']
    report.append(f"**风险等级**: {'★' * risk['risk_level']} (共5级)")
    report.append(f"**就医建议**: {risk['urgency']}")
    report.append("**具体建议**:")
    for i, rec in enumerate(risk['recommendations'], 1):
        report.append(f"{i}. {rec}")
    report.append("")
    
    # 治疗方案
    plan = result['diagnosis']
    report.append("### 💊 治疗方案")
    report.append("**推荐检查项目**:")
    for i, exam in enumerate(plan.get('examinations', []), 1):
        report.append(f"{i}. {exam}")
    report.append("")
    
    report.append("**用药建议**:")
    for i, med in enumerate(plan.get('medications', []), 1):
        report.append(f"{i}. {med}")
    report.append("")
    
    report.append("**生活建议**:")
    for i, life in enumerate(plan.get('lifestyle', []), 1):
        report.append(f"{i}. {life}")
    
    # 免责声明
    report.append("\n> ⚠️ 免责声明：本结果仅供参考，不替代专业医疗诊断")
    
    return "\n".join(report)

def visualize_knowledge(symptoms: List[str]) -> str:
    """可视化与症状相关的知识图谱"""
    if not graph_kb:
        return "知识图谱尚未初始化"
    
    if not symptoms:
        return "未提取到症状，无法生成知识图谱"
    
    # 生成可视化文件
    graph_kb.visualize(
        filename="symptom_related_kg.html",
        highlight_entities=symptoms,
        max_nodes=50
    )
    
    return f"已生成与症状相关的知识图谱：<a href='symptom_related_kg.html' target='_blank'>查看图谱</a>"

def create_interface():
    """创建Gradio界面（支持自然语言+图像输入）"""
    with gr.Blocks(title="医疗智能诊断系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🏥 医疗智能诊断系统
        ## 多模态输入（自然语言+图像）
        
        请用自然语言描述您的不适，或上传医疗图像（如皮肤病灶、检查报告），系统将自动分析并提供建议。
        """)
        
        # 存储诊断报告的状态变量
        diagnosis_report_state = gr.State(value="")
        
        with gr.Row():
            with gr.Column(scale=3):
                # 自然语言输入（核心：无需手动输入症状）
                user_input = gr.Textbox(
                    label="症状描述（自然语言）", 
                    placeholder="例如：我最近三天发烧、咳嗽，胸口有点痛，还伴有乏力...", 
                    lines=4
                )
                
                # 图像输入
                image_input = gr.Image(
                    label="上传医疗图像（可选）",
                    type="filepath",
                    sources=["upload", "webcam"],
                    height=250
                )
                
                # 病史输入
                medical_history = gr.Textbox(
                    label="病史（可选）", 
                    placeholder="如有基础疾病或过敏史，请在此说明...", 
                    lines=2
                )
                
                # 诊断报告输入（用于生成可视化）
                report_input = gr.Textbox(
                    label="诊断报告（可选，用于生成相关知识图谱）", 
                    placeholder="可粘贴诊断报告内容...", 
                    lines=4
                )
                
                # 操作按钮
                with gr.Row():
                    diagnose_btn = gr.Button("🔍 获取诊断建议", variant="primary")
                    visualize_btn = gr.Button("📊 可视化知识图谱（基于症状）")
                    visualize_from_report_btn = gr.Button("📈 从诊断报告生成图谱")
                    clear_btn = gr.Button("🗑️ 清除输入", variant="secondary")
                
                # 结果输出
                result_output = gr.Markdown(label="诊断结果")
            
            with gr.Column(scale=1):
                status_output = gr.Textbox(
                    label="系统状态", 
                    value="初始化中...", 
                    interactive=False
                )
        
        # 诊断逻辑（核心：自动提取症状）
        def diagnose(text_input, image_path, history_text):
            status = "正在提取症状并分析..."
            
            # 1. 多模态提取症状（自然语言+图像→症状列表）
            symptoms = ernie_client.extract_symptoms_from_multimodal(
                text=text_input, 
                image_path=image_path
            )
            
            if not symptoms:
                return "无法从输入中提取到有效症状，请重新描述或上传清晰图像。", "提取失败", ""
            
            # 2. 处理诊断
            result = process_query(symptoms, history_text)
            
            # 3. 添加图像分析结果（如果有）
            if image_path:
                try:
                    result["image_analysis"] = ernie_client.medical_image_analysis(image_path)
                except:
                    result["image_analysis"] = "图像分析失败"
            
            formatted = format_result(result)
            return formatted, "诊断完成", result.get("full_report", "")
        
        # 从诊断报告生成知识图谱
        def visualize_from_report(report_text):
            if not report_text:
                return "请输入诊断报告内容", "未提供报告"
            
            viz_result = visualization_agent.generate_visualization_from_report(report_text)
            return viz_result, "图谱生成完成"
        
        # 知识图谱可视化（基于症状）
        def visualize(text_input, image_path):
            status = "正在提取症状并生成知识图谱..."
            
            # 提取症状
            symptoms = ernie_client.extract_symptoms_from_multimodal(
                text=text_input, 
                image_path=image_path
            )
            
            if not symptoms:
                return "无法提取症状，无法生成知识图谱。", "提取失败"
            
            viz_result = visualize_knowledge(symptoms)
            return viz_result, "知识图谱生成完成"
        
        # 清除输入
        def clear_inputs():
            return None, "", "", "", "# 诊断结果将显示在这里", "已清除输入", ""
        
        # 绑定事件
        diagnose_btn.click(
            fn=diagnose,
            inputs=[user_input, image_input, medical_history],
            outputs=[result_output, status_output, report_input]
        )
        
        visualize_btn.click(
            fn=visualize,
            inputs=[user_input, image_input],
            outputs=[result_output, status_output]
        )
        
        visualize_from_report_btn.click(
            fn=visualize_from_report,
            inputs=[report_input],
            outputs=[result_output, status_output]
        )
        
        clear_btn.click(
            fn=clear_inputs,
            outputs=[image_input, user_input, medical_history, report_input, result_output, status_output, diagnosis_report_state]
        )
        
        # 快捷键支持（回车提交）
        user_input.submit(
            fn=diagnose,
            inputs=[user_input, image_input, medical_history],
            outputs=[result_output, status_output, report_input]
        )
        
        # 系统初始化
        demo.load(
            fn=init_system,
            inputs=None,
            outputs=[status_output]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
