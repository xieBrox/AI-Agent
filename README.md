# ERNIE-Ai-Agent-Applications

<div align="center">
  <a href="README_zh.md">
    <img src="https://img.shields.io/badge/语言-中文-blue?style=for-the-badge" alt="中文版">
  </a>
  <a href="README.md">
    <img src="https://img.shields.io/badge/Language-English-red?style=for-the-badge" alt="English">
  </a>
</div>

<br>

> A practical project collection for ERNIE large language model applications, covering medical AI agents, retrieval-augmented generation (RAG), multimodal model deployment, and invoice automation.

## 🌟 Project Overview

This repository focuses on **ERNIE large language model application development**, providing four end-to-end practical projects to help developers master ERNIE's implementation in real-world scenarios. Leveraging ERNIE's strengths in Chinese understanding, multimodal processing, and knowledge integration, these projects demonstrate how to build intelligent systems with industrial value.

### 🎯 What You'll Learn
- **Multimodal Applications**: Processing text and image inputs for medical scenarios
- **Knowledge Enhancement**: Building retrieval-augmented generation systems with vector databases
- **Local Deployment**: Deploying ERNIE-4.5-VL (Vision-Language) model for offline use
- **OCR + LLM Integration**: Automating invoice processing with PPOCR and ERNIE
- **Practical Tools**: Integrating Gradio interfaces, ChromaDB, and multi-agent architectures

## 📁 Project Collection

### 🏥 Medical-Agent (`Medical-Agent/`)
An intelligent medical consultation system powered by ERNIE, enabling AI-assisted symptom analysis and health recommendations.

| Core Function | Technical Implementation | Key Features |
|---------------|--------------------------|--------------|
| Symptom Analysis | ERNIE-based NLP, multi-agent architecture | Extracts key symptoms from text descriptions and medical images |
| Risk Assessment | Knowledge retrieval + ERNIE reasoning | Generates 1-5 risk levels and urgency recommendations |
| Treatment Planning | Medical knowledge base integration | Provides examination, medication, and lifestyle suggestions |
| User Interface | Gradio interactive interface | Supports text input and image uploads (e.g., X-rays, skin photos) |

**Key Files**:
- `agents.py`: Multi-agent system (symptom parser, knowledge retriever, diagnosis agent)
- `ernie_client.py`: ERNIE model interaction (text generation, image-text analysis)
- `knowledge_base.py`: Medical knowledge storage and retrieval with ChromaDB
- `main.gradio.py`: User-friendly visual interface


### 🔍 RAG-Tutorial (`RAG-Tutorial/`)
A step-by-step guide to building retrieval-augmented generation systems with ERNIE, focusing on document processing and knowledge retrieval.

| Core Function | Technical Implementation | Key Features |
|---------------|--------------------------|--------------|
| Document Processing | Intelligent text chunking | Splits documents while preserving semantic integrity (character/token double verification) |
| Knowledge Base Construction | ChromaDB vector database | Supports multiple embedding functions (default, SentenceTransformer) |
| Efficient Retrieval | Similarity search optimization | Batch processing and structured data storage (JSONL format) |
| Augmented Generation | ERNIE + retrieved context | Enhances response accuracy with knowledge grounding |

**Key Files**:
- `document_processor.py`: Text chunking and preprocessing (jieba for tokenization)
- `chroma_builder.py`: Vector database operations (data loading, querying, statistics)
- `requirements.txt`: Dependencies for document processing and database management


### 🖼️ ERNIE-4.5-VL Local Deployment (`ERNIE-4.5-VL-Local-Deployment-Tutorial/`)
A comprehensive tutorial for deploying ERNIE-4.5-VL (Vision-Language) model locally, enabling multimodal applications without relying on cloud services.

| Core Function | Technical Implementation | Key Features |
|---------------|--------------------------|--------------|
| Local Service Setup | FastAPI/UVicorn deployment | Establishes RESTful API for model interactions |
| Multimodal Processing | Image-text joint understanding | Supports image input (Base64 encoding) and cross-modal tasks |
| Environment Configuration | Hardware optimization guidance | GPU acceleration and resource management tips |
| Practical Examples | Sample inference code | Demos for image description, visual question answering, etc. |


### 🔖 PPOCR-invoice-automation (`PPOCR-invoice-automation/`)
An intelligent invoice processing system integrating PPOCR (Baidu's OCR toolkit) and ERNIE, enabling automated extraction, verification, and structured storage of invoice information.

| Core Function | Technical Implementation | Key Features |
|---------------|--------------------------|--------------|
| Invoice Recognition | PPOCR + image preprocessing | High-precision extraction of text from various invoice types (fapiao, receipt, etc.) |
| Information Structuring | ERNIE-based NLP | Converts unstructured text to structured data (amount, date, payer, item details) |
| Data Validation | Rule engine + ERNIE reasoning | Verifies logical consistency (e.g., amount format, date validity) |
| Batch Processing | Asynchronous task queue | Supports bulk invoice processing with progress tracking |
| Export Function | Multi-format output | Generates Excel/JSON reports for financial systems integration |

**Key Files**:
- `invoice_processor.py`: OCR recognition and text extraction pipeline
- `ernie_validator.py`: ERNIE-based information verification and correction
- `data_exporter.py`: Structured data export to various formats
- `main.py`: Command-line interface for batch processing


## 🛠️ Tech Stack

| Category | Components | Description |
|----------|------------|-------------|
| **Core Model** | ERNIE Series | ERNIE large language model (text understanding, multimodal processing) |
| **Vector Database** | ChromaDB | Knowledge storage and similarity search |
| **OCR Tool** | PPOCR | Baidu's OCR toolkit for text extraction from images |
| **Interface** | Gradio | Interactive visual interface for user interaction |
| **Backend** | FastAPI/UVicorn | API service deployment for model interactions |
| **Text Processing** | jieba | Chinese word segmentation for token counting |
| **Image Handling** | Pillow, Base64 | Image encoding and preprocessing |
| **Development Tools** | OpenAI Client | Compatible interface for ERNIE model calls |


## 🚀 Quick Start

### Environment Requirements
- **Python**: 3.8+
- **Dependencies**: See each project's `requirements.txt`
- **Optional**: GPU with sufficient VRAM (for local model deployment)


### Installation

```bash
# Clone the repository
git clone https://github.com/xieBrox/ERNIE-AI-Applications.git
cd ERNIE-AI-Applications

# Install dependencies for Medical-Agent
cd Medical-Agent
pip install -r requirements.txt

# Install dependencies for RAG-Tutorial
cd ../RAG-Tutorial
pip install -r requirements.txt

# Install dependencies for PPOCR-invoice-automation
cd ../PPOCR-invoice-automation
pip install -r requirements.txt

# Follow deployment guide for ERNIE-4.5-VL
cd ../ERNIE-4.5-VL-Local-Deployment-Tutorial
# Refer to the tutorial for model download and environment setup
```


### Run Projects

1. **Medical-Agent**
```bash
cd Medical-Agent
python main.gradio.py
# Access the interface via the provided local URL
```

2. **RAG-Tutorial**
```bash
cd RAG-Tutorial
# Process documents
python document_processor.py
# Build knowledge base
python chroma_builder.py
```

3. **ERNIE-4.5-VL Deployment**
```bash
# Follow the step-by-step guide in the project directory
cd ERNIE-4.5-VL-Local-Deployment-Tutorial
# Start local service according to the tutorial
```

4. **PPOCR-invoice-automation**
```bash
cd PPOCR-invoice-automation
# Process single invoice
python main.py --file path/to/invoice.jpg
# Batch process invoices in a folder
python main.py --folder path/to/invoice_folder --output results.xlsx
```


## 📊 Project Features

| Feature | Description |
|---------|-------------|
| **Practical Orientation** | All projects focus on real-world scenarios, with complete workflows from input to output |
| **ERNIE Optimization** | Tailored for ERNIE's strengths in Chinese understanding and multimodal processing |
| **Modular Design** | Clear separation of components (model interaction, data processing, interface) for easy extension |
| **Detailed Logging** | Comprehensive logging systems for debugging and performance analysis |
| **User-Friendly** | Gradio interfaces and detailed comments lower the barrier to use |


## 🤝 Contributing

We welcome contributions to improve these projects! Here are ways to contribute:

- 🐛 Report bugs or issues in the project
- 💡 Suggest new features or improvements for ERNIE applications
- 📝 Enhance documentation or add tutorials
- 🔧 Submit code improvements (e.g., optimization, new functions)


### Contribution Process
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

   
## 📄 License

This project is licensed under the [Apache 2.0 License](LICENSE). You are free to use, modify, and distribute the code, with attribution to the original repository.

## 📞 Contact

- **Repository Maintainer**: xieBrox
- **Issue Tracking**: Use GitHub Issues for questions or problems
- **Project Link**: [ERNIE-AI-Agent-Applications](https://github.com/xieBrox/ERNIE-AI-Agent-Applications)

---

⭐ **If you find these projects helpful, please give us a Star! Your support encourages further development.**

🚀 **Start building ERNIE-based applications today!**
