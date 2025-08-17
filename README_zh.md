# ERNIE-AI-Agent

<div align="center">
  <a href="README_zh.md">
    <img src="https://img.shields.io/badge/语言-中文-blue?style=for-the-badge" alt="中文版">
  </a>
  <a href="README.md">
    <img src="https://img.shields.io/badge/Language-English-red?style=for-the-badge" alt="English">
  </a>
</div>

<br>

> 基于ERNIE大模型的实用应用集合，涵盖医疗AI代理、检索增强生成（RAG）和多模态模型部署。

## 🌟 项目概述

本仓库专注于**ERNIE大模型应用开发**，提供三个端到端的实战项目，帮助开发者掌握ERNIE在实际场景中的落地方法。依托ERNIE在中文理解、多模态处理和知识融合方面的优势，这些项目展示了如何构建具有产业价值的智能系统。

### 🎯 你将学到
- **多模态应用**：处理医疗场景中的文本和图像输入
- **知识增强**：结合向量数据库构建检索增强生成系统
- **本地部署**：离线部署ERNIE-4.5-VL（视觉-语言）模型
- **实用工具**：集成Gradio界面、ChromaDB和多代理架构

## 📁 项目集合

### 🏥 医疗AI代理（`Medical-Agent/`）
基于ERNIE的智能医疗咨询系统，支持AI辅助的症状分析和健康建议。

| 核心功能 | 技术实现 | 主要特点 |
|---------|---------|---------|
| 症状解析 | 基于ERNIE的自然语言处理、多代理架构 | 从文本描述和医疗图像中提取关键症状 |
| 风险评估 | 知识检索+ERNIE推理 | 生成1-5级风险等级和紧急程度建议 |
| 治疗方案 | 医疗知识库集成 | 提供检查、用药和生活方式建议 |
| 用户界面 | Gradio交互式界面 | 支持文本输入和图像上传（如X光片、皮肤照片） |

**关键文件**：
- `agents.py`：多代理系统（症状解析器、知识检索器、诊断代理）
- `ernie_client.py`：ERNIE模型交互（文本生成、图文分析）
- `knowledge_base.py`：基于ChromaDB的医疗知识存储与检索
- `main.gradio.py`：用户友好的可视化界面

### 🔍 RAG教程（`RAG-Tutorial/`）
基于ERNIE构建检索增强生成系统的分步指南，重点关注文档处理和知识检索。

| 核心功能 | 技术实现 | 主要特点 |
|---------|---------|---------|
| 文档处理 | 智能文本分块 | 保留语义完整性的文档拆分（字符/ token双重验证） |
| 知识库构建 | ChromaDB向量数据库 | 支持多种嵌入函数（默认函数、SentenceTransformer） |
| 高效检索 | 相似度搜索优化 | 批量处理和结构化数据存储（JSONL格式） |
| 增强生成 | ERNIE+检索上下文 | 通过知识 grounding 提升响应准确性 |

**关键文件**：
- `document_processor.py`：文本分块与预处理（使用jieba分词）
- `chroma_builder.py`：向量数据库操作（数据加载、查询、统计）
- `requirements.txt`：文档处理和数据库管理的依赖项

### 🖼️ ERNIE-4.5-VL本地部署（`ERNIE-4.5-VL-Local-Deployment-Tutorial/`）
ERNIE-4.5-VL（视觉-语言）模型本地部署的综合教程，无需依赖云服务即可实现多模态应用。

| 核心功能 | 技术实现 | 主要特点 |
|---------|---------|---------|
| 本地服务搭建 | FastAPI/UVicorn部署 | 建立用于模型交互的RESTful API |
| 多模态处理 | 图文联合理解 | 支持图像输入（Base64编码）和跨模态任务 |
| 环境配置 | 硬件优化指南 | GPU加速和资源管理技巧 |
| 实战示例 | 推理示例代码 | 图像描述、视觉问答等场景演示 |

## 🛠️ 技术栈

| 类别 | 组件 | 说明 |
|------|------|------|
| **核心模型** | ERNIE系列 | ERNIE大语言模型（文本理解、多模态处理） |
| **向量数据库** | ChromaDB | 知识存储与相似度搜索 |
| **交互界面** | Gradio | 用于用户交互的可视化界面 |
| **后端服务** | FastAPI/UVicorn | 模型交互的API服务部署 |
| **文本处理** | jieba | 用于token计数的中文分词工具 |
| **图像处理** | Pillow、Base64 | 图像编码与预处理 |
| **开发工具** | OpenAI Client | 兼容ERNIE模型调用的接口 |

## 🚀 快速开始

### 环境要求
- **Python**：3.8+
- **依赖项**：参见各项目的`requirements.txt`
- **可选**：具有足够显存的GPU（用于本地模型部署）

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/xieBrox/ERNIE-AI-Applications.git
cd ERNIE-AI-Applications

# 安装Medical-Agent依赖
cd Medical-Agent
pip install -r requirements.txt

# 安装RAG-Tutorial依赖
cd ../RAG-Tutorial
pip install -r requirements.txt

# 部署ERNIE-4.5-VL（参考项目内教程）
cd ../ERNIE-4.5-VL-Local-Deployment-Tutorial
# 按照教程进行模型下载和环境配置
```

### 运行项目

1. **医疗AI代理**
```bash
cd Medical-Agent
python main.gradio.py
# 通过提供的本地URL访问界面
```

2. **RAG教程**
```bash
cd RAG-Tutorial
# 处理文档
python document_processor.py
# 构建知识库
python chroma_builder.py
```

3. **ERNIE-4.5-VL部署**
```bash
# 参考项目目录中的分步指南
cd ERNIE-4.5-VL-Local-Deployment-Tutorial
# 按照教程启动本地服务
```

## 📊 项目特点

| 特点 | 说明 |
|------|------|
| **实战导向** | 所有项目聚焦真实场景，包含从输入到输出的完整工作流 |
| **ERNIE优化** | 针对ERNIE在中文理解和多模态处理的优势进行定制 |
| **模块化设计** | 组件分离清晰（模型交互、数据处理、界面），便于扩展 |
| **详细日志** | 完善的日志系统，方便调试和性能分析 |
| **易用性** | Gradio界面和详细注释降低使用门槛 |

## 🤝 贡献指南

我们欢迎任何形式的贡献来改进这些项目！贡献方式包括：

- 🐛 报告项目中的bug或问题
- 💡 提出ERNIE应用的新功能或改进建议
- 📝 完善文档或添加教程
- 🔧 提交代码改进（如优化、新功能）

### 贡献流程
1. Fork本仓库
2. 创建功能分支（`git checkout -b feature/YourFeature`）
3. 提交更改（`git commit -m 'Add YourFeature'`）
4. 推送到分支（`git push origin feature/YourFeature`）
5. 打开Pull Request

## 📄 许可证

本项目基于[Apache 2.0许可证](LICENSE)授权。您可以自由使用、修改和分发代码，但请保留对原始仓库的引用。

## 📞 联系方式

- **仓库维护者**：xieBrox
- **问题追踪**：使用GitHub Issues提交问题或疑问
- **项目链接**：[ERNIE-AI-Applications](https://github.com/xieBrox/ERNIE-AI-Agent-Applications)

---

⭐ **如果这些项目对您有帮助，请给我们一个Star！您的支持将鼓励我们持续开发。**

🚀 **立即开始构建基于ERNIE的应用吧！**
