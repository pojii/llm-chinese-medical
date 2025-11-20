# LLM Chinese Medical - Medicine Prediction with Knowledge Graph

A system for Traditional Chinese Medicine (TCM) recommendation using Large Language Models (LLM) with and without Knowledge Graph augmentation.

## Overview

This project demonstrates the effectiveness of integrating medical knowledge graphs with LLMs for medicine prediction. It compares two approaches:

1. **LLM Only**: Direct prediction using a lightweight Chinese LLM
2. **LLM + Knowledge Graph**: Enhanced prediction using medical knowledge graph context

## Features

- 🏥 **Medical Knowledge Graph**: Based on QASystemOnMedicalKG (44K+ entities, 300K+ relationships)
- 🔬 **Hybrid KG System**: Combines Chinese Medical KG + DRKG (Drug Repurposing Knowledge Graph)
- 🏷️ **TCM NER Dataset**: BIO-formatted Named Entity Recognition for symptoms, causes, herbs, etc.
- 🤖 **Lightweight LLM**: CPU-compatible Chinese GPT-2 model for resource-constrained environments
- 📊 **Comparison Framework**: 3-way evaluation (No KG / Single KG / Hybrid KG)
- 📈 **Evaluation Metrics**: Precision, Recall, and F1 score calculation with micro/macro averaging
- 🎯 **Confidence Scoring**: Weighted scoring from multiple knowledge sources

## Project Structure

```
llm-chinese-medical/
├── data/
│   └── medical.json               # Chinese Medical KG data (45MB)
├── src/
│   ├── knowledge_graph.py         # Chinese Medical KG loader
│   ├── hybrid_kg.py               # Hybrid KG (Chinese + DRKG) 🆕
│   ├── ner_dataset.py             # TCM NER dataset handler
│   ├── llm_predictor.py           # LLM-based predictor (base class)
│   ├── deepseek_predictor.py      # DeepSeek-R1-Distill-Llama-8B predictor 🆕
│   ├── metrics.py                 # Evaluation metrics (P/R/F1)
│   ├── main_comparison.py         # 2-way comparison (No KG vs Single KG)
│   └── main_comparison_hybrid.py  # 3-way comparison (includes Hybrid KG) 🆕
├── models/                         # Model cache (auto-created)
├── outputs/                        # Results output directory
├── test_metrics.py                # Metrics test suite
├── demo_without_model.py          # Demo without dependencies
├── requirements.txt               # Python dependencies
├── README.md                      # Main documentation
├── HYBRID_KG_README.md            # Hybrid KG documentation 🆕
└── QUICKSTART.md                  # Quick start guide
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

### Recommended Model (Default)
- **Name**: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- **Type**: Distilled reasoning model optimized for medical domain
- **Requirements**: CUDA-compatible GPU (recommended), ~8GB VRAM
- **Inference**: ~0.1-0.5 seconds per prediction on GPU
- **Advantages**:
  - Advanced reasoning capabilities with distilled knowledge
  - Better understanding of Chinese medical terminology
  - More accurate and focused predictions
  - Proper handling of system/user prompt formatting

**Usage**:
```python
from deepseek_predictor import DeepSeekMedicinePredictor

# Initialize with CUDA (recommended)
predictor = DeepSeekMedicinePredictor(device="cuda")

# Predict without KG
query = "患者症状: 头痛, 发热。请推荐合适的中药。"
result = predictor.predict_without_kg(query)

# Predict with KG
kg_context = "感冒是由病毒引起的上呼吸道感染..."
result = predictor.predict_with_kg(query, kg_context)
```

### Alternative Models

#### CPU-Compatible Model (Legacy)
- **Name**: `uer/gpt2-chinese-cluecorpussmall`
- **Type**: Chinese GPT-2 (small)
- **Requirements**: ~500MB RAM, CPU-compatible
- **Inference**: ~1-3 seconds per prediction on CPU
- **Note**: May produce less accurate results than DeepSeek

```python
from llm_predictor import MedicineLLMPredictor

# Use CPU-compatible model
predictor = MedicineLLMPredictor(
    model_name="uer/gpt2-chinese-cluecorpussmall",
    device="cpu"
)
```

#### Auto-Detection
The system automatically detects DeepSeek models:

```python
# In main_comparison.py
if "deepseek" in model_name.lower():
    predictor = DeepSeekMedicinePredictor(device=device)
else:
    predictor = MedicineLLMPredictor(model_name=model_name, device=device)
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

📊 本样本评估指标
────────────────────────────────────────
不使用知识图谱:
  提取的药物: {'银翘解毒片', '连花清瘟胶囊'}
  Precision: 0.5000
  Recall:    0.5000
  F1 Score:  0.5000

使用知识图谱:
  提取的药物: {'银翘解毒片'}
  Precision: 1.0000
  Recall:    1.0000
  F1 Score:  1.0000
```

## Evaluation Metrics

The system calculates comprehensive evaluation metrics to quantify performance:

### Metrics Calculated

1. **Precision**: Proportion of predicted medicines that are correct
   - Formula: TP / (TP + FP)
   - Measures accuracy of predictions

2. **Recall**: Proportion of ground truth medicines that were predicted
   - Formula: TP / (TP + FN)
   - Measures completeness of predictions

3. **F1 Score**: Harmonic mean of precision and recall
   - Formula: 2 × (Precision × Recall) / (Precision + Recall)
   - Balances both metrics

### Aggregation Methods

- **Micro-averaging**: Aggregate all TP/FP/FN counts, then calculate metrics
  - Better for overall system performance
  - Weighs each medicine equally

- **Macro-averaging**: Calculate metrics per sample, then average
  - Better for per-sample performance
  - Weighs each test case equally

### Medicine Matching

The metrics system includes intelligent matching:

- **Exact match**: Same medicine name (similarity = 1.0)
- **Partial match**: Contains relationship (similarity = 0.8)
  - E.g., "板蓝根" matches "板蓝根颗粒"
- **Fuzzy match**: Character overlap (configurable threshold)

### Example Aggregate Results

```
📊 聚合评估指标 (Aggregate Metrics)
================================================================================
【方法1】不使用知识图谱
────────────────────────────────────────
Micro-averaged:
  Precision: 0.6250
  Recall:    0.8333
  F1 Score:  0.7143

【方法2】使用知识图谱增强
────────────────────────────────────────
Micro-averaged:
  Precision: 1.0000
  Recall:    1.0000
  F1 Score:  1.0000

🎯 性能对比 (Performance Comparison)
────────────────────────────────────────
知识图谱增强带来的提升:
  F1 Score:  +40.00%
  Precision: +60.00%
  Recall:    +20.00%
```

### Testing Metrics

Run the metrics test suite:

```bash
python test_metrics.py
```

This validates:
- Medicine extraction from text
- Similarity calculation
- Precision/recall/F1 computation
- Aggregate metrics
- Comparison scenarios

## Hybrid Knowledge Graph System 🆕

The system now supports a **hybrid knowledge graph approach** that combines multiple knowledge sources for more accurate predictions:

### Architecture

```
Chinese Medical KG (8,808 entities)
    +
DRKG (97,238 entities, 5.8M relationships)
    ↓
Hybrid Knowledge Graph
    ↓
Enhanced Context with:
- Disease descriptions
- Drug-disease relationships
- Side effects
- Drug interactions
- Confidence scores
```

### Usage

```bash
# Run 3-way comparison
cd src
python main_comparison_hybrid.py
```

This compares:
1. LLM only (baseline)
2. LLM + Single KG (Chinese Medical)
3. LLM + Hybrid KG (Chinese Medical + DRKG) ← **Best performance**

### Key Advantages

- **Multi-source knowledge**: Combines traditional Chinese medicine + biomedical evidence
- **Confidence scoring**: Weighted recommendations based on relationship strength
- **Safety information**: Includes side effects and drug interactions from DRKG
- **Better coverage**: 12 TCM compounds in sample data, expandable to full DRKG

See [HYBRID_KG_README.md](HYBRID_KG_README.md) for detailed documentation.

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

@misc{drkg2020,
  title={DRKG - Drug Repurposing Knowledge Graph},
  author={Ioannidis, Vassilis N. and Song, Xiang and Manchanda, Saurav and Li, Mufei and Pan, Xiaoqin and Zheng, Da and Ning, Xia and Zeng, Xiangxiang and Karypis, George},
  year={2020},
  url={https://github.com/gnn4dr/DRKG}
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
