import os
import sys
import json
import logging
import tempfile
import gradio as gr
from datetime import datetime
from invoice_processor import InvoiceProcessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app.log'
)
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp_uploads")
os.makedirs(TEMP_DIR, exist_ok=True)
tempfile.tempdir = TEMP_DIR

HOST = "0.0.0.0"
PORT = "7000"
BASE_URL = f"http://{HOST}:{PORT}/v1"
API_KEY = "null"


def process_invoice_step(image, step, current_state=None):
    """分步处理发票（移除API参数，使用固定配置）"""
    # 1. 基础校验
    if image is None:
        msg = "请上传发票图片"
        logger.warning(msg)
        return None, msg, None, gr.update(interactive=False), gr.update(interactive=False)
    
    # 2. 检查文件是否存在
    if not os.path.exists(image):
        msg = f"上传失败：文件不存在（路径：{image}）"
        logger.error(msg)
        return None, msg, None, gr.update(interactive=False), gr.update(interactive=False)
    
    # 3. 检查文件大小（限制10MB以内）
    file_size = os.path.getsize(image) / (1024 * 1024)
    if file_size > 10:
        msg = f"上传失败：文件过大（{file_size:.2f}MB，最大支持10MB）"
        logger.warning(msg)
        return None, msg, None, gr.update(interactive=False), gr.update(interactive=False)
    
    try:
        # 初始化处理器
        processor = InvoiceProcessor()
        
        # 分步处理逻辑
        if step == "ocr":
            logger.info(f"开始OCR处理：{os.path.basename(image)}")
            result = processor.process_invoice_basic(image)
            return (
                result,
                json.dumps(result.get("summary", {}), ensure_ascii=False, indent=2),
                None,
                gr.update(interactive=True),
                gr.update(interactive=False)
            )
            
        elif step == "company_info":
            if not current_state:
                msg = "请先进行OCR识别"
                return None, msg, None, gr.update(interactive=False), gr.update(interactive=False)
            
            logger.info(f"获取企业信息：{current_state.get('filename')}")
            company_info = processor.get_company_information(current_state)
            current_state["company_info"] = company_info
            return (
                current_state,
                json.dumps(current_state.get("summary", {}), ensure_ascii=False, indent=2),
                json.dumps(company_info, ensure_ascii=False, indent=2),
                gr.update(interactive=True),
                gr.update(interactive=True)
            )
            
        elif step == "analysis":
            if not current_state or "company_info" not in current_state:
                msg = "请先获取企业信息"
                return None, msg, None, gr.update(interactive=False), gr.update(interactive=False)
            
            logger.info(f"生成分析报告：{current_state.get('filename')}")
            analysis = processor.generate_analysis_report(current_state)
            current_state["analysis_report"] = analysis
            return (
                current_state,
                json.dumps(current_state.get("summary", {}), ensure_ascii=False, indent=2),
                json.dumps(analysis, ensure_ascii=False, indent=2),
                gr.update(interactive=True),
                gr.update(interactive=True)
            )
    
    except Exception as e:
        msg = f"处理失败：{str(e)}"
        logger.error(f"处理异常：{msg}", exc_info=True)
        return None, msg, None, gr.update(interactive=False), gr.update(interactive=False)


def process_multiple_invoices(files):
    """批量处理发票（移除API参数）"""
    if not files:
        return "请上传发票图片", None
    
    # 检查所有文件是否有效
    valid_files = []
    for f in files:
        if not os.path.exists(f.name):
            logger.error(f"批量处理失败：文件不存在（{f.name}）")
            return f"上传失败：文件 {os.path.basename(f.name)} 不存在", None
        if os.path.getsize(f.name) / (1024 * 1024) > 10:
            logger.error(f"批量处理失败：文件过大（{f.name}）")
            return f"上传失败：文件 {os.path.basename(f.name)} 超过10MB", None
        valid_files.append(f.name)
    
    try:
        # 初始化处理器（使用固定配置）
        processor = InvoiceProcessor()
        
        logger.info(f"开始批量处理：共{len(valid_files)}张发票")
        result = processor.process_multiple_invoices(valid_files)
        
        # 格式化结果
        summary = result["summary"]
        summary_text = f"""
        处理完成！
        - 总计处理: {summary['total_count']} 张发票
        - 成功: {summary['success_count']} 张
        - 失败: {summary['fail_count']} 张
        - 总金额: ¥{summary['total_amount']:.2f}
        - Excel文件: {os.path.basename(summary['excel_path'])} （保存路径：{summary['excel_path']}）
        """
        return summary_text, result["excel_path"]
    
    except Exception as e:
        msg = f"批量处理失败：{str(e)}"
        logger.error(msg, exc_info=True)
        return msg, None


def create_web_interface():
    """创建Gradio Web界面（移除API配置输入）"""
    with gr.Blocks(
        title="AI发票智能识别系统", 
        theme=gr.themes.Soft(),
        css="""
        .gr-button-primary { font-size: 16px; padding: 10px 20px; }
        .gr-markdown h3 { color: #2c3e50; }
        .upload-container { border: 2px dashed #3498db; border-radius: 8px; padding: 20px; }
        """
    ) as interface:
        gr.Markdown("# 🧾 AI发票智能识别系统")
        
        # 保存当前处理状态
        current_state = gr.State(None)
        
        with gr.Tabs():
            # 单张处理标签页
            with gr.Tab("单张处理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 上传发票图片")
                        input_image = gr.Image(
                            label="点击或拖放图片到此处",
                            type="filepath",
                            sources="upload",
                            height=250,
                            elem_classes="upload-container"
                        )
                        
                        # 移除API配置输入框（使用固定配置）
                        gr.Markdown("### 系统配置")
                        gr.Textbox(
                            label="大模型服务地址",
                            value=BASE_URL,
                            interactive=False,
                            info="固定配置：本地服务"
                        )
                        gr.Textbox(
                            label="API密钥",
                            value=API_KEY,
                            type="password",
                            interactive=False,
                            info="固定配置：null"
                        )
                        
                        with gr.Row():
                            process_button = gr.Button("1. 处理发票（OCR识别）", variant="primary")
                            company_button = gr.Button("2. 获取企业信息", interactive=False)
                            analysis_button = gr.Button("3. 生成分析报告", interactive=False)
                    
                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.Tab("📊 发票信息"):
                                summary_output = gr.JSON(label="结构化识别结果")
                            
                            with gr.Tab("🏢 企业信息"):
                                company_output = gr.JSON(label="购买方/销售方信息")
                            
                            with gr.Tab("📋 分析报告"):
                                analysis_output = gr.JSON(label="交易风险与分析")
                
                # 绑定按钮事件
                process_button.click(
                    fn=process_invoice_step,
                    inputs=[input_image, gr.State("ocr"), current_state],
                    outputs=[current_state, summary_output, company_output, company_button, analysis_button],
                    show_progress=True
                )
                
                company_button.click(
                    fn=process_invoice_step,
                    inputs=[input_image, gr.State("company_info"), current_state],
                    outputs=[current_state, summary_output, company_output, company_button, analysis_button],
                    show_progress=True
                )
                
                analysis_button.click(
                    fn=process_invoice_step,
                    inputs=[input_image, gr.State("analysis"), current_state],
                    outputs=[current_state, summary_output, analysis_output, company_button, analysis_button],
                    show_progress=True
                )

            # 批量处理标签页
            with gr.Tab("批量处理"):
                with gr.Column():
                    gr.Markdown("### 批量上传发票图片")
                    input_files = gr.File(
                        label="选择多张图片（支持JPG/PNG）",
                        file_count="multiple",
                        file_types=["image"],
                        height=150,
                        elem_classes="upload-container"
                    )
                    
                    # 移除批量处理的API配置
                    gr.Markdown("### 系统配置")
                    gr.Textbox(
                        label="大模型服务地址",
                        value=BASE_URL,
                        interactive=False,
                        info="固定配置：本地服务"
                    )
                    
                    batch_process_button = gr.Button("批量处理并生成Excel", variant="primary")
                    batch_output = gr.Textbox(label="处理结果", lines=6)
                    excel_output = gr.File(label="下载Excel报告")
                
                # 批量处理事件
                batch_process_button.click(
                    fn=process_multiple_invoices,
                    inputs=[input_files],
                    outputs=[batch_output, excel_output],
                    show_progress=True
                )
        
        gr.Markdown("""
        ### 使用说明
        1. **单张处理流程**：
           - 上传发票图片（支持JPG/PNG，最大10MB）
           - 依次点击三个按钮完成：OCR识别 → 企业信息获取 → 分析报告生成
        
        2. **批量处理流程**：
           - 选择多张发票图片（建议不超过10张）
           - 点击"批量处理"按钮，自动生成Excel报告
        
        3. 系统配置：
           - 大模型服务地址：http://0.0.0.0:7000/v1
           - API密钥：null（固定配置）
        """)
    
    return interface


if __name__ == "__main__":
    logger.info("启动发票识别系统...")
    interface = create_web_interface()
    # 启动服务（支持外部访问）
    interface.launch()