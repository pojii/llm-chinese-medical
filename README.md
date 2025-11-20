# LLM Chinese Medical - Medicine Recommender System with Knowledge Graph

A comprehensive medical recommender system using Large Language Models (LLM) with Knowledge Graph augmentation. Provides ranked drug/treatment recommendations with rigorous evaluation metrics.

## Overview

This project demonstrates the effectiveness of integrating medical knowledge graphs with LLMs for medicine recommendation. It provides two complementary systems:

### 1. **Medicine Recommender System** (NEW) 🎯
Generates ranked lists of drug/treatment recommendations with evaluation using:
- **Precision@K** (K=5,10,20,50): Accuracy of top-K recommendations
- **Recall@K**: Coverage of relevant drugs in top-K
- **MAP@K** (Mean Average Precision): Overall recommendation quality
- **MRR@K** (Mean Reciprocal Rank): Ranking effectiveness

### 2. **Named Entity Recognition (NER) System** (Original)
Extracts medical entities from clinical text:
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
│   ├── knowledge_graph.py          # Chinese Medical KG loader
│   ├── hybrid_kg.py                # Hybrid KG (Chinese + DRKG)
│   ├── ner_dataset.py              # TCM NER dataset handler
│   ├── recommender_dataset.py      # Recommender dataset (symptom→drugs) 🆕
│   ├── recommender_predictor.py    # Recommender API wrapper 🆕
│   ├── recommender_metrics.py      # Recommender metrics (P@K, R@K, MAP, MRR) 🆕
│   ├── deepseek_api_predictor.py   # DeepSeek API predictor ⭐ (recommended)
│   ├── deepseek_predictor.py       # DeepSeek local model predictor (GPU)
│   ├── llm_predictor.py            # Base LLM predictor class
│   ├── metrics.py                  # NER evaluation metrics (P/R/F1)
│   ├── main_recommender.py         # Recommender evaluation 🆕
│   ├── main_comparison.py          # 2-way NER comparison (No KG vs Single KG)
│   └── main_comparison_hybrid.py   # 3-way NER comparison (includes Hybrid KG)
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

#### Medicine Recommender System (Recommended) 🆕

Run the recommender evaluation with ranking metrics:

```bash
# Set API key
export DEEPSEEK_API_KEY='your-api-key-here'

# Run recommender evaluation
cd src
python main_recommender.py
```

This will:
1. Load the medical knowledge graph (15,250+ samples from medical.json)
2. Initialize the recommender predictor
3. Generate ranked drug recommendations for medical queries
4. Evaluate using Precision@K, Recall@K, MAP@K, MRR@K (K=5,10,20,50)
5. Compare with and without KG augmentation
6. Save results to `outputs/recommender_results.json`

**Configuration options in `main_recommender.py`:**
```python
config = {
    'kg_path': '../data/medical.json',
    'model': 'deepseek-chat',
    'num_samples': 10,           # Number of test samples
    'query_type': None,          # 'disease', 'symptom', or None for all
    'min_drugs': 3,              # Minimum ground truth drugs
    'max_recommendations': 50     # Max recommendations per query
}
```

#### NER System (Original)

Run the NER comparison experiment:

```bash
cd src
python main_comparison.py
```

This will:
1. Load the medical knowledge graph
2. Initialize the LLM predictor
3. Run entity extraction on TCM cases
4. Compare results with and without KG augmentation
5. Save results to `outputs/comparison_results.json`

### Recommender System Components 🆕

#### 1. Recommender Dataset

```python
from recommender_dataset import MedicalRecommenderDataset

# Load dataset (extracts symptom→drugs and disease→drugs pairs)
dataset = MedicalRecommenderDataset("../data/medical.json")
print(f"Loaded {len(dataset)} samples")

# Get a sample
sample = dataset.get_sample(0)
print(f"Query Type: {sample['query_type']}")  # 'disease' or 'symptom'
print(f"Query: {sample['query']}")
print(f"Ground Truth: {sample['ground_truth']}")

# Format as LLM query
query = dataset.format_query(sample)
```

#### 2. Recommender Predictor

```python
from recommender_predictor import MedicalRecommenderPredictor

predictor = MedicalRecommenderPredictor(model="deepseek-chat")

# Recommend without KG (returns ranked list)
recommendations = predictor.recommend_without_kg(query, top_k=10)
print(f"Top 10 drugs: {recommendations}")

# Recommend with KG
recommendations_kg = predictor.recommend_with_kg(query, kg_context, top_k=10)
```

#### 3. Recommender Metrics

```python
from recommender_metrics import RecommenderMetrics

metrics = RecommenderMetrics()

# Evaluate single prediction
recommended = ["药A", "药B", "药C", "药D", "药E"]
relevant = {"药A", "药B", "药F"}

result = metrics.evaluate_single(recommended, relevant)
print(f"Precision@5: {result[5]['precision']}")
print(f"Recall@5: {result[5]['recall']}")
print(f"MAP@5: {result[5]['map']}")
print(f"MRR@5: {result[5]['mrr']}")

# Accumulate over multiple samples
metrics.update(recommended, relevant)
aggregate = metrics.get_aggregate_metrics()
```

### NER System Components (Original)

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

### Recommended: DeepSeek API ⭐ (No GPU Required)
- **Type**: Cloud-based API service
- **Requirements**: DeepSeek API key only (no GPU needed)
- **Inference**: ~0.5-2 seconds per prediction
- **Advantages**:
  - No GPU or heavy dependencies required
  - Advanced reasoning capabilities
  - Better understanding of Chinese medical terminology
  - Cost-effective for most use cases
  - Automatic scaling and updates

**Setup**:

1. Get your DeepSeek API key from [https://platform.deepseek.com](https://platform.deepseek.com)

2. Set the API key as environment variable:
```bash
export DEEPSEEK_API_KEY='your-api-key-here'
```

3. Install dependencies:
```bash
pip install openai numpy
```

**Usage**:
```python
from deepseek_api_predictor import DeepSeekAPIPredictor

# Initialize with API (reads DEEPSEEK_API_KEY from environment)
predictor = DeepSeekAPIPredictor(model="deepseek-chat")

# Predict without KG
query = "患者症状: 头痛, 发热。请推荐合适的中药。"
result = predictor.predict_without_kg(query)

# Predict with KG
kg_context = "感冒是由病毒引起的上呼吸道感染..."
result = predictor.predict_with_kg(query, kg_context)
```

**Running the comparison:**
```bash
# Set API key
export DEEPSEEK_API_KEY='your-api-key-here'

# Run comparison
cd src
python main_comparison.py
```

### Alternative: Local Model (For Offline Use)

If you need to run without internet connection, you can use local models:

**Option 1: DeepSeek R1 Distill Llama 8B (GPU Required)**
- **Name**: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- **Requirements**: CUDA GPU with 2-8GB VRAM
- **Advantages**: Best accuracy for local deployment

```python
from deepseek_predictor import DeepSeekMedicinePredictor

# Standard (8GB VRAM)
predictor = DeepSeekMedicinePredictor(device="cuda")

# 4-bit quantization (2GB VRAM)
predictor = DeepSeekMedicinePredictor(device="cuda", load_in_4bit=True)
```

**Option 2: Chinese GPT-2 (CPU Compatible)**
- **Name**: `uer/gpt2-chinese-cluecorpussmall`
- **Requirements**: ~500MB RAM, CPU-compatible
- **Note**: Lower accuracy than DeepSeek

```python
from llm_predictor import MedicineLLMPredictor

predictor = MedicineLLMPredictor(
    model_name="uer/gpt2-chinese-cluecorpussmall",
    device="cpu"
)
```

**Switching between API and Local:**
```python
# In main_comparison.py config
config = {
    'use_api': True,  # Set to False for local model
    'model_name': 'deepseek-chat'  # Or local model name
}
```

## Results

### Recommender System Results 🆕

The recommender system outputs comprehensive ranking metrics:

```
================================================================================
Sample 1/10
================================================================================

Query Type: symptom
Query: 患者出现以下症状：发热、咳嗽、咽痛、流鼻涕，请推荐适合的药物或治疗方法。
Disease: 感冒
Ground Truth (6 drugs): ['连花清瘟胶囊', '板蓝根颗粒', '银翘解毒片', ...]

────────────────────────────────────────────────────────────────────────────────
【Method 1】Recommendation WITHOUT Knowledge Graph
────────────────────────────────────────────────────────────────────────────────
Top 10 Recommendations: ['连花清瘟胶囊', '感冒清热颗粒', '板蓝根颗粒', ...]
Inference Time: 1.23s

────────────────────────────────────────────────────────────────────────────────
【Method 2】Recommendation WITH Knowledge Graph
────────────────────────────────────────────────────────────────────────────────
Retrieved KG Context:
疾病名称: 感冒
症状: 发热、咳嗽、咽痛...
推荐药物: 连花清瘟胶囊、板蓝根颗粒...

Top 10 Recommendations: ['连花清瘟胶囊', '板蓝根颗粒', '银翘解毒片', ...]
Inference Time: 1.45s

────────────────────────────────────────────────────────────────────────────────
📊 Sample Metrics
────────────────────────────────────────────────────────────────────────────────

Without KG:
  P@5:   0.4000
  R@5:   0.3333
  MAP@5: 0.4667
  MRR@5: 1.0000

With KG:
  P@5:   0.6000
  R@5:   0.5000
  MAP@5: 0.7333
  MRR@5: 1.0000

================================================================================
📊 Aggregate Recommender Metrics
================================================================================
Total samples: 10

【Method 1】WITHOUT Knowledge Graph
────────────────────────────────────────────────────────────────────────────────

Metric          @5           @10          @20          @50
────────────────────────────────────────────────────────────────────────────────
Precision       0.3800       0.2400       0.1450       0.0680
Recall          0.3210       0.4560       0.5890       0.7120
MAP             0.4123       0.4456       0.4678       0.4832
MRR             0.7234       0.7234       0.7234       0.7234

【Method 2】WITH Knowledge Graph
────────────────────────────────────────────────────────────────────────────────

Metric          @5           @10          @20          @50
────────────────────────────────────────────────────────────────────────────────
Precision       0.5200       0.3100       0.1850       0.0820
Recall          0.4780       0.5940       0.7120       0.8340
MAP             0.6234       0.6512       0.6734       0.6891
MRR             0.8945       0.8945       0.8945       0.8945

────────────────────────────────────────────────────────────────────────────────
🎯 Performance Comparison
────────────────────────────────────────────────────────────────────────────────

Improvement from Knowledge Graph Enhancement:

Metric          @5           @10          @20          @50
────────────────────────────────────────────────────────────────────────────────
Precision       +36.84%      +29.17%      +27.59%      +20.59%
Recall          +48.91%      +30.26%      +20.88%      +17.13%
MAP             +51.19%      +46.16%      +43.97%      +42.62%
MRR             +23.65%      +23.65%      +23.65%      +23.65%
```

**Key Advantages of Recommender Metrics:**
- **Precision@K**: Shows accuracy at different cut-offs (top-5, top-10, etc.)
- **Recall@K**: Measures coverage of relevant drugs
- **MAP@K**: Evaluates overall ranking quality
- **MRR@K**: Assesses how quickly relevant items appear
- **Real-world applicability**: Users can choose from top-K suggestions

### NER System Results (Original)

The NER system outputs comparison results showing:

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

- [x] ~~Implement evaluation metrics (precision, recall, F1)~~ ✅
- [x] ~~Add recommender system with ranking metrics (P@K, R@K, MAP, MRR)~~ ✅
- [ ] Add vector search for better KG retrieval
- [ ] Support for larger/better Chinese medical LLMs
- [ ] Interactive web interface
- [ ] Integration with more medical knowledge sources (e.g., TCM databases)
- [ ] Fine-tuning on TCM-specific data
- [ ] Personalized recommendations based on patient history
- [ ] Multi-modal medical data integration (images, lab results)

## Contact

For questions or issues, please open an issue on GitHub.
