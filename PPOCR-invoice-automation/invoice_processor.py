import os
import re
import time
import logging
import pandas as pd
import json
from typing import List, Dict, Any, Optional
from paddleocr import PaddleOCR
import openai
from datetime import datetime
from agents import MultiAgentSystem, BASE_URL

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='invoice_processing.log'
)
logger = logging.getLogger(__name__)


class InvoiceProcessor:
    def __init__(self, lang='ch'):
        """初始化OCR模型和大模型客户端（固定配置）"""
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang)
        logger.info("OCR模型初始化完成")
        
        # 初始化大模型客户端
        self.client = openai.Client(
            base_url=BASE_URL,
            api_key="null"
        )
        
        # 初始化多Agent系统
        self.multi_agent = MultiAgentSystem()
        
        self.keyword_mapping = {
            "发票号码": ["发票号码", "发票号"],
            "金额": ["金额", "合计金额", "价税合计", "总金额"],
            "日期": ["日期", "开票日期"],
            "购买方名称": ["购买方名称", "客户名称", "购方名称"],
            "销售方名称": ["销售方名称", "销售单位", "销方名称"]
        }
        
        self.ocr_cache = {}
    
    def extract_text_from_image(self, img_path: str, region=None) -> str:
        """从图片中提取文字（保持不变）"""
        if img_path in self.ocr_cache:
            logger.info(f"使用缓存的OCR结果: {img_path}")
            return self.ocr_cache[img_path]
            
        if not os.path.exists(img_path):
            logger.error(f"图片文件不存在: {img_path}")
            raise FileNotFoundError(f"图片文件不存在: {img_path}")
            
        logger.info(f"正在从 {img_path} 提取文本...")
        try:
            result = self.ocr.predict(img_path)
            if result and len(result) > 0 and "rec_texts" in result[0]:
                text = '\n'.join([t.strip() for t in result[0]["rec_texts"] if t.strip()])
                text = text.replace('\r', '').replace('  ', ' ')
                
                logger.info(f"文本提取完成，共 {len(text)} 个字符")
                self.ocr_cache[img_path] = text
                return text
            raise ValueError("OCR未识别到有效文本")
        except Exception as e:
            logger.error(f"文本提取失败: {str(e)}")
            raise
    
    def recognize_text(self, image_path: str) -> List[str]:
        """识别图片中的所有文本（保持不变）"""
        full_text = self.extract_text_from_image(image_path)
        return [line.strip() for line in full_text.split('\n') if line.strip()]
    
    def extract_key_info(self, texts: List[str]) -> Dict[str, str]:
        """从文本中提取关键信息（保持不变）"""
        extracted_info = {key: None for key in self.keyword_mapping.keys()}
        
        for text in texts:
            for key, keywords in self.keyword_mapping.items():
                if any(kw in text for kw in keywords):
                    if ":" in text or "：" in text:
                        value = re.split(r'[:：]', text)[-1].strip()
                        extracted_info[key] = value
                    else:
                        for kw in keywords:
                            if kw in text:
                                value = text.replace(kw, "").strip()
                                extracted_info[key] = value
                                break
        
        return extracted_info
    
    def extract_structured_data(self, ocr_text: str) -> Dict[str, str]:
        """调用大模型提取结构化数据（修改为流式调用）"""
        if not ocr_text.strip():
            raise ValueError("输入文本为空，无法提取数据")
            
        prompt = f"""
        你是专业的税务发票解析专家，请严格按照以下要求提取信息：
        
        1. 提取字段及说明：
           - invoice_number: 发票号码（优先取"No"或"发票号码"后的数字，如存在重复以清晰的主号码为准）
           - invoice_date: 开票日期（格式统一为"YYYY年MM月DD日"，补全缺失的"0"）
           - total_amount: 价税合计金额（取大写金额对应的数字，如"壹万圆整"对应10000.00）
           - tax_amount: 税额（明确标注的税额数值）
           - taxable_amount: 不含税金额（明确标注的金额数值）
           - buyer: 购买方全称（从"购买方"或"名 称："下方提取）
           - buyer_tax_id: 购买方纳税人识别号
           - seller: 销售方全称（从"销售方"或"名 称："下方提取）
           - seller_tax_id: 销售方纳税人识别号
           - invoice_type: 发票类型（根据内容判断，如含"融资租赁"则标注"融资租赁发票"）
           - item_name: 货物或服务名称（主项目名称）
           - specification: 规格型号（如有）
           - tax_rate: 税率（如13%）
        
        2. 特殊规则：
           - 金额需保留两位小数，大写金额需转换为数字（如"壹万圆整"→10000.00）
           - 若存在多个相同字段（如多个号码），取最清晰完整的一个
           - 无法识别的字段用空字符串""，不要留空或用null
        
        3. 输出格式：仅返回JSON，不添加任何解释文字
        {{
            "invoice_number": "",
            "invoice_date": "",
            "total_amount": "",
            "tax_amount": "",
            "taxable_amount": "",
            "buyer": "",
            "buyer_tax_id": "",
            "seller": "",
            "seller_tax_id": "",
            "invoice_type": "",
            "item_name": "",
            "specification": "",
            "tax_rate": ""
        }}
        
        发票文本内容：
        {ocr_text}
        """
        
        try:
            logger.info("正在调用大模型提取结构化数据...")
            # 调用大模型
            response = self.client.chat.completions.create(
                model="null",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                stream=True  # 启用流式
            )
            
            # 收集流式响应内容
            result_text = ""
            for chunk in response:
                if chunk.choices[0].delta and chunk.choices[0].delta.content:
                    result_text += chunk.choices[0].delta.content
            
            # 解析结果
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result_json = json.loads(json_match.group(0))
            else:
                result_json = json.loads(result_text)
                
            logger.info("结构化数据提取完成")
            return result_json
            
        except json.JSONDecodeError:
            logger.error(f"无法解析模型返回的JSON: {result_text}")
            raise
        except Exception as e:
            logger.error(f"大模型调用失败: {str(e)}")
            raise
    
    def generate_summary(self, ocr_texts: List[str]) -> Dict[str, Any]:
        """兼容原有接口"""
        full_text = "\n".join(ocr_texts)
        try:
            return self.extract_structured_data(full_text)
        except Exception as e:
            logger.error(f"生成摘要失败: {str(e)}")
            return {"error": str(e)}
    
    def process_invoice_basic(self, image_path: str) -> Dict[str, Any]:
        """基础OCR处理"""
        try:
            full_text = self.extract_text_from_image(image_path)
            texts = self.recognize_text(image_path)
            key_info = self.extract_key_info(texts)
            
            result = {
                "filename": os.path.basename(image_path),
                "ocr_texts": texts,
                "full_text": full_text,
                "extracted_info": key_info,
                "processing_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "成功"
            }
            
            if full_text.strip():
                result["summary"] = self.extract_structured_data(full_text)
            else:
                result["summary"] = {"error": "OCR未识别到有效文本"}
            
            return result
            
        except Exception as e:
            logger.error(f"基础处理失败: {e}")
            return {
                "status": "失败",
                "error": str(e),
                "processing_time": time.strftime("%Y-%m-%d %H:%M:%S")
            }

    def get_company_information(self, invoice_data: Dict[str, Any]) -> Dict[str, Any]:
        """获取企业信息"""
        try:
            summary = invoice_data.get("summary", {})
            seller_name = summary.get("seller", "")
            buyer_name = summary.get("buyer", "")
            
            seller_info = self.multi_agent.company_agent.get_company_info(seller_name) if seller_name else None
            buyer_info = self.multi_agent.company_agent.get_company_info(buyer_name) if buyer_name else None
            
            return {
                "seller": seller_info,
                "buyer": buyer_info,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            logger.error(f"获取企业信息失败: {e}")
            return {
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

    def generate_analysis_report(self, invoice_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成分析报告"""
        try:
            summary = invoice_data.get("summary", {})
            company_info = invoice_data.get("company_info", {})
            
            analysis = self.multi_agent.analysis_agent.generate_report(
                summary,
                company_info
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"生成分析报告失败: {e}")
            return {
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
    
    def process_invoice(self, image_path: str) -> Dict[str, Any]:
        """处理单张发票"""
        try:
            filename = os.path.basename(image_path)
            full_text = self.extract_text_from_image(image_path)
            texts = self.recognize_text(image_path)
            key_info = self.extract_key_info(texts)
            
            result = {
                "filename": filename,
                "ocr_texts": texts,
                "full_text": full_text,
                "extracted_info": key_info,
                "processing_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "成功"
            }
            
            if full_text.strip():
                result["summary"] = self.extract_structured_data(full_text)
            else:
                result["summary"] = {"error": "OCR未识别到有效文本"}
                return result
            
            if result["status"] == "成功":
                analysis_result = self.multi_agent.process_invoice_with_analysis(result)
                result.update({
                    "company_info": analysis_result.get("company_info", {}),
                    "analysis_report": analysis_result.get("analysis_report", {})
                })
            
            return result
            
        except Exception as e:
            logger.error(f"处理发票 {image_path} 出错: {e}")
            return {
                "filename": filename if 'filename' in locals() else os.path.basename(image_path),
                "error": str(e),
                "status": "失败",
                "processing_time": time.strftime("%Y-%m-%d %H:%M:%S")
            }
    
    def process_multiple_invoices(self, image_paths: List[str], output_dir: str = "output") -> Dict[str, Any]:
        """批量处理发票"""
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        all_analysis = []
        
        batch_size = 5
        total = len(image_paths)
        
        for i in range(0, total, batch_size):
            batch = image_paths[i:i+batch_size]
            for j, image_path in enumerate(batch, 1):
                idx = i + j
                logger.info(f"处理第 {idx}/{total} 张发票: {image_path}")
                try:
                    result = self.process_invoice(image_path)
                    
                    invoice_data = {
                        "文件名": result.get("filename", ""),
                        "处理状态": result.get("status", "失败"),
                        "处理时间": result.get("processing_time", ""),
                        "发票号码": "",
                        "开票日期": "",
                        "购买方": "",
                        "销售方": "",
                        "金额": "",
                        "发票类型": "",
                        "风险等级": "",
                        "备注": ""
                    }
                    
                    if result.get("status") == "成功":
                        summary = result.get("summary", {})
                        analysis = result.get("analysis_report", {})
                        company_info = result.get("company_info", {})
                        
                        invoice_data.update({
                            "发票号码": summary.get("invoice_number", ""),
                            "开票日期": summary.get("invoice_date", ""),
                            "购买方": summary.get("buyer", ""),
                            "销售方": summary.get("seller", ""),
                            "金额": summary.get("total_amount", ""),
                            "发票类型": summary.get("invoice_type", ""),
                            "风险等级": analysis.get("risk_assessment", {}).get("risk_level", "未知"),
                            "备注": analysis.get("summary", "")
                        })
                        
                        if analysis:
                            all_analysis.append({
                                "发票信息": invoice_data,
                                "企业信息": company_info,
                                "分析报告": analysis
                            })
                    else:
                        invoice_data["备注"] = result.get("error", "处理失败")
                    
                    results.append(invoice_data)
                    
                except Exception as e:
                    logger.error(f"处理发票 {image_path} 时出错: {e}")
                    results.append({
                        "文件名": os.path.basename(image_path),
                        "处理状态": "失败",
                        "处理时间": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "备注": str(e)
                    })
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = os.path.join(output_dir, f"发票处理结果_{timestamp}.xlsx")
        
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df = pd.DataFrame(results)
                df.to_excel(writer, sheet_name='发票数据', index=False)
                
                if all_analysis:
                    analysis_df = pd.DataFrame([{
                        "发票号码": item["发票信息"]["发票号码"],
                        "交易分析": item["分析报告"].get("transaction_analysis", ""),
                        "企业分析": item["分析报告"].get("company_analysis", ""),
                        "风险评估": str(item["分析报告"].get("risk_assessment", "")),
                        "建议": str(item["分析报告"].get("recommendations", ""))
                    } for item in all_analysis])
                    analysis_df.to_excel(writer, sheet_name='详细分析', index=False)
            
            logger.info(f"处理完成，结果已保存至: {excel_path}")
        except Exception as e:
            logger.error(f"生成Excel报告失败: {e}")
            excel_path = None
        
        try:
            total_amount = 0.0
            for r in results:
                if r.get("金额"):
                    try:
                        amount_str = str(r["金额"]).replace("¥", "").replace(",", "").strip()
                        total_amount += float(amount_str)
                    except (ValueError, TypeError):
                        continue
        except Exception as e:
            logger.warning(f"计算总金额时出错: {e}")
            total_amount = 0.0
        
        summary = {
            "total_count": total,
            "success_count": len([r for r in results if r["处理状态"] == "成功"]),
            "fail_count": len([r for r in results if r["处理状态"] == "失败"]),
            "total_amount": round(total_amount, 2),
            "excel_path": excel_path,
            "processing_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return {
            "excel_path": excel_path,
            "summary": summary,
            "detailed_analysis": all_analysis
        }