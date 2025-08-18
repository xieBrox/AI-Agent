import os
import re
import json
import logging
import requests
import random
import time
from bs4 import BeautifulSoup
from urllib.parse import quote
from typing import Dict, Any, List, Optional
from datetime import datetime
import openai
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='agents.log'
)
logger = logging.getLogger(__name__)

# Fixed large model service configuration
HOST = "0.0.0.0"
PORT = "7000"
BASE_URL = f"http://{HOST}:{PORT}/v1"


class CompanyInfoAgent:
    """Company Information Retrieval Agent"""

    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Connection": "keep-alive"
        }
        self.proxies = None
        if os.environ.get("USE_PROXY", "false").lower() == "true":
            proxy_url = os.environ.get("PROXY_URL")
            if proxy_url:
                self.proxies = {
                    "http": proxy_url,
                    "https": proxy_url
                }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(2) + wait_fixed(3),
        retry=retry_if_exception_type((requests.exceptions.RequestException, requests.exceptions.Timeout))
    )
    def get_company_info(self, company_name: str) -> Optional[Dict[str, Any]]:
        if not company_name or len(company_name.strip()) < 2:
            logger.warning("Company name is too short to query")
            return None

        time.sleep(random.uniform(1, 3))
        url = f"https://www.tianyancha.com/search?key={quote(company_name)}"

        try:
            response = requests.get(
                url,
                headers=self.headers,
                proxies=self.proxies,
                timeout=15,
                allow_redirects=True
            )
            response.raise_for_status()

            if "login" in response.url or "注册" in response.text:
                logger.warning("Access restricted, login may be required")
                return {"warning": "Company information retrieval restricted, Tianyancha login may be required"}

            soup = BeautifulSoup(response.text, 'html.parser')
            data = {}

            first_item = soup.find('div', class_=re.compile(r'index_search-item__W7iG_'))
            if not first_item:
                logger.info(f"No information found for company {company_name}")
                return None

            name_div = first_item.find('div', class_=re.compile(r'index_name__qEdWi'))
            data['company_name'] = ''.join(name_div.stripped_strings) if name_div else 'N/A'

            tags = first_item.find_all('span', class_='index_tag-item__9dloe')
            data['tags'] = ''.join([tag.get_text(strip=True) for tag in tags]) if tags else 'N/A'

            info_divs = first_item.find_all('div', class_='index_info-col__UVcZb')
            for div in info_divs:
                text = div.get_text(strip=True)
                if '：' in text:
                    label, value = text.split('：', 1)
                    label_mapping = {
                        '法定代表人': 'legal_representative',
                        '注册资本': 'registered_capital',
                        '成立日期': 'establishment_date',
                        '统一社会信用代码': 'unified_social_credit_code',
                        '经营状态': 'business_status',
                        '企业类型': 'company_type'
                    }
                    if label in label_mapping:
                        data[label_mapping[label]] = value

            contact_divs = first_item.find_all('div', class_='index_contact-col__7AboU')
            for div in contact_divs:
                label = div.get_text(strip=True).split('：')[0]
                value_span = div.find('span', class_='index_value__Pl0Nh')
                if label and value_span:
                    data[label] = value_span.get_text(strip=True)

            risk_divs = soup.find_all('div', class_='index_risk-count__zyBjB')
            if len(risk_divs) >= 2:
                data['self_risk'] = risk_divs[0].get_text(strip=True)
                data['peripheral_risk'] = risk_divs[1].get_text(strip=True)
                data['risk_summary'] = f"{data['self_risk']}, {data['peripheral_risk']}"

            logger.info(f"Successfully retrieved information for company {company_name}")
            return data

        except Exception as e:
            logger.error(f"Error retrieving information for company {company_name}: {e}")
            if isinstance(e, (requests.exceptions.RequestException, requests.exceptions.Timeout)):
                raise
            return {"error": f"Failed to retrieve company information: {str(e)}"}


class AnalysisReportAgent:
    """Analysis Report Generation Agent"""

    def __init__(self):
        # Initialize large model client
        self.client = openai.Client(
            base_url=BASE_URL,
            api_key="null"  # Fixed as null
        )

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_fixed(3)
    )
    def generate_report(self, invoice_data: Dict[str, Any], company_info: Dict[str, Any]) -> Dict[str, Any]:
        try:
            prompt = """
            Please generate a detailed analysis report based on the following invoice information and company information.

            Invoice information:
            {}

            Company information:
            {}

            Please analyze from the following aspects:
            1. Basic transaction information (amount, time, type of goods/services, etc.)
            2. Company background analysis (company size, industry position, credit status, etc.)
            3. Risk assessment (based on company risk information and transaction characteristics)
            4. Recommendations and reminders (matters needing attention for this transaction)

            Please return in JSON format, including the following fields:
            {{
                "transaction_analysis": "Transaction analysis",
                "company_analysis": "Company analysis",
                "risk_assessment": {{
                    "risk_level": "low/medium/high",
                    "risk_points": ["Risk point 1", "Risk point 2"],
                    "details": "Detailed description"
                }},
                "recommendations": ["Recommendation 1", "Recommendation 2"],
                "summary": "Summary"
            }}
            """.format(
                json.dumps(invoice_data, ensure_ascii=False, indent=2),
                json.dumps(company_info, ensure_ascii=False, indent=2)
            )

            # Call large model (streaming response, model fixed as null)
            response = self.client.chat.completions.create(
                model="null",  # Fixed as null
                messages=[
                    {"role": "user", "content": prompt}
                ],
                stream=True  # Enable streaming
            )

            # Collect streaming response content
            result_text = ""
            for chunk in response:
                if chunk.choices[0].delta and chunk.choices[0].delta.content:
                    result_text += chunk.choices[0].delta.content

            # Parse result
            try:
                report = json.loads(result_text)
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    try:
                        report = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        report = {
                            "error": "Unable to parse report content",
                            "raw_text": json_match.group(0)
                        }
                else:
                    report = {
                        "error": "Unable to parse report content",
                        "raw_text": result_text
                    }

            logger.info("Successfully generated analysis report")
            return report

        except Exception as e:
            logger.error(f"Error generating analysis report: {e}")
            raise


class MultiAgentSystem:
    """Multi-Agent Collaborative System"""

    def __init__(self):
        self.company_agent = CompanyInfoAgent()
        self.analysis_agent = AnalysisReportAgent()  # No longer needs external api_key and base_url

    def process_invoice_with_analysis(self, invoice_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            summary = invoice_data.get("summary", {})
            seller_name = summary.get("seller", "")
            buyer_name = summary.get("buyer", "")

            seller_info = self.company_agent.get_company_info(seller_name) if seller_name else None
            buyer_info = self.company_agent.get_company_info(buyer_name) if buyer_name else None

            analysis = self.analysis_agent.generate_report(
                invoice_data,
                {
                    "seller": seller_info,
                    "buyer": buyer_info
                }
            )

            return {
                "invoice_data": invoice_data,
                "company_info": {
                    "seller": seller_info,
                    "buyer": buyer_info
                },
                "analysis_report": analysis,
                "processing_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

        except Exception as e:
            logger.error(f"Error in multi-agent processing: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }