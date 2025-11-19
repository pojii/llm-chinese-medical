# LLM Chinese Medical - Medicine Prediction with Knowledge Graph

A system for Traditional Chinese Medicine (TCM) recommendation using Large Language Models (LLM) with and without Knowledge Graph augmentation.

## Overview

This project demonstrates the effectiveness of integrating medical knowledge graphs with LLMs for medicine prediction. It compares two approaches:

1. **LLM Only**: Direct prediction using a lightweight Chinese LLM
2. **LLM + Knowledge Graph**: Enhanced prediction using medical knowledge graph context

## Features

- 🏥 **Medical Knowledge Graph**: Based on QASystemOnMedicalKG (44K+ entities, 300K+ relationships)
- 🏷️ **TCM NER Dataset**: BIO-formatted Named Entity Recognition for symptoms, causes, herbs, etc.
- 🤖 **Lightweight LLM**: CPU-compatible Chinese GPT-2 model for resource-constrained environments
- 📊 **Comparison Framework**: Side-by-side evaluation of KG-augmented vs. non-augmented predictions

## Project Structure

```
llm-chinese-medical/
├── data/
│   └── medical.json          # Medical knowledge graph data
├── src/
│   ├── knowledge_graph.py    # KG loader and query system
│   ├── ner_dataset.py        # TCM NER dataset handler
│   ├── llm_predictor.py      # LLM-based predictor
│   └── main_comparison.py    # Main comparison script
├── models/                    # Model cache (auto-created)
├── outputs/                   # Results output directory
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd llm-chinese-medical
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download data

The medical knowledge graph data is automatically included. The dataset is from:
- **Source**: [QASystemOnMedicalKG](https://github.com/zhihao-chen/QASystemOnMedicalKG)
- **Citation**:
  ```bibtex
  @misc{chen2021qasystem,
    title={QASystemOnMedicalKG},
    author={Chen, Zhihao},
    year={2021},
    publisher={GitHub},
    journal={GitHub repository},
    howpublished={\url{https://github.com/zhihao-chen/QASystemOnMedicalKG}}
  }
  ```

## Usage

### Quick Start

Run the comparison experiment:

```bash
cd src
python main_comparison.py
```

This will:
1. Load the medical knowledge graph
2. Initialize the LLM predictor
3. Run predictions on sample TCM cases
4. Compare results with and without KG augmentation
5. Save results to `outputs/comparison_results.json`

### Individual Components

#### 1. Knowledge Graph

```python
from knowledge_graph import MedicalKnowledgeGraph

kg = MedicalKnowledgeGraph("./data/medical.json")

# Search for relevant context
context = kg.search_relevant_context("头痛发热", top_k=3)
print(context)

# Get disease information
disease_info = kg.get_disease_info("感冒")
```

#### 2. NER Dataset

```python
from ner_dataset import TCMNERDataset

dataset = TCMNERDataset()  # Uses sample data

# Get a sample
sample = dataset.get_sample(0)
print(f"Text: {sample['text']}")

# Extract entities
entities = dataset.extract_entities(sample)
print(f"Symptoms: {entities['SYM']}")
print(f"Herbs: {entities['HER']}")
```

#### 3. LLM Predictor

```python
from llm_predictor import MedicineLLMPredictor

predictor = MedicineLLMPredictor(device="cpu")

# Predict without KG
query = "患者出现头痛发热症状。请推荐合适的中药。"
result = predictor.predict_without_kg(query)
print(result)

# Predict with KG
kg_context = "感冒是由病毒引起的上呼吸道感染..."
result = predictor.predict_with_kg(query, kg_context)
print(result)
```

## NER Label System

The TCM NER dataset uses BIO tagging format:

- **B-SYM, I-SYM**: Symptoms (症状)
- **B-CAU, I-CAU**: Causes (病因)
- **B-HER, I-HER**: Herbs/Medicine (药物)
- **B-PRE, I-PRE**: Prescriptions (处方)
- **B-EFF, I-EFF**: Effects (功效)
- **O**: Other tokens

## Model Configuration

### Default Model
- **Name**: `uer/gpt2-chinese-cluecorpussmall`
- **Type**: Chinese GPT-2 (small)
- **Requirements**: ~500MB RAM, CPU-compatible
- **Inference**: ~1-3 seconds per prediction on CPU

### Alternative Models

You can use other lightweight Chinese models:

```python
# Example: Using a different model
predictor = MedicineLLMPredictor(
    model_name="THUDM/chatglm-6b-int4",  # Quantized model
    device="cpu"
)
```

## Results

The system outputs comparison results showing:

- Original text and extracted entities
- Ground truth medicine labels
- Predictions with and without KG
- Inference time for each method
- Retrieved KG context

Example output:

```
Sample 1/5
─────────────────────────────────────
原文: 患者出现头痛发热症状，建议服用银翘解毒片进行治疗。
查询: 患者症状: 头痛, 发热。请推荐合适的中药。
真实标注的药物: ['银翘解毒片']

【方法1】不使用知识图谱的LLM预测
预测结果: 银翘解毒片、连花清瘟胶囊 (基于LLM直接推荐)
推理时间: 0.15秒

【方法2】使用知识图谱增强的LLM预测
检索到的知识图谱上下文:
疾病名称: 感冒
描述: 感冒是由病毒引起的上呼吸道感染...
预测结果: 银翘解毒片 (基于知识图谱推荐)
推理时间: 0.18秒
```

## Key Findings

1. **Knowledge Graph Benefits**:
   - Provides domain-specific medical knowledge
   - Enhances LLM understanding of medical terminology
   - Improves recommendation accuracy and explainability

2. **LLM-Only Limitations**:
   - Relies solely on pre-trained knowledge
   - May generate inaccurate or irrelevant recommendations
   - Lacks professional medical knowledge support

3. **Resource Efficiency**:
   - Uses lightweight CPU-compatible models
   - Suitable for resource-constrained environments
   - Acceptable inference speed for practical applications

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{chen2021qasystem,
  title={QASystemOnMedicalKG},
  author={Chen, Zhihao},
  year={2021},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished={\url{https://github.com/zhihao-chen/QASystemOnMedicalKG}}
}
```

## License

This project is for educational and research purposes.

## Disclaimer

⚠️ **Medical Disclaimer**: This system is for research and educational purposes only. The medicine recommendations are generated by AI models and should NOT be used for actual medical diagnosis or treatment. Always consult qualified healthcare professionals for medical advice.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Future Improvements

- [ ] Add more sophisticated retrieval methods (e.g., vector search)
- [ ] Implement evaluation metrics (precision, recall, F1)
- [ ] Support for larger/better Chinese medical LLMs
- [ ] Interactive web interface
- [ ] Integration with more medical knowledge sources
- [ ] Fine-tuning on TCM-specific data

## Contact

For questions or issues, please open an issue on GitHub.
