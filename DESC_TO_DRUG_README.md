# Drug Recommendation from Disease Description

## Overview

ระบบแนะนำยาจาก **คำอธิบายโรค** (disease description) โดยไม่ใช้ Knowledge Graph (baseline model)

### Task Definition

- **Input**: `desc` field - คำอธิบายโรคภาษาจีน
- **Output**: `recommand_drug` - รายการยาที่แนะนำ (ranked list)
- **Evaluation**: Precision@K, Recall@K สำหรับ K = 5, 10, 15, 20, 30

### Why This Task?

1. **Real-world application**: แพทย์อ่านอาการ/คำอธิบาย แล้วแนะนำยา
2. **Challenging**: ต้องทำนายจาก 3,800 ยาที่เป็นไปได้
3. **Baseline for KG enhancement**: เพื่อเปรียบเทียบกับ KG-enhanced model

## Dataset Statistics

จาก `medical.json`:

```
Total samples: 7,645 (diseases with drug recommendations)
Unique drugs: 3,800
Average drugs per disease: 7.78
Min drugs: 1
Max drugs: 23
Median drugs: 6
Std drugs: 5.56
```

### Top 10 Most Frequent Drugs

```
1. 乳酸左氧氟沙星片 (Levofloxacin Lactate Tablets) - 403 occurrences
2. 盐酸左氧氟沙星胶囊 (Levofloxacin HCl Capsules) - 382 occurrences
3. 消癌平片 (Xiaoaiping Tablets) - 270 occurrences
4. 头孢氨苄胶囊 (Cephalexin Capsules) - 252 occurrences
5. 依托红霉素片 (Erythromycin Tablets) - 238 occurrences
6. 氨苄西林胶囊 (Ampicillin Capsules) - 229 occurrences
7. 阿奇霉素片 (Azithromycin Tablets) - 223 occurrences
8. 强力康颗粒 (Qianglikang Granules) - 218 occurrences
9. 注射用头孢唑林钠 (Cefazolin Sodium Injection) - 213 occurrences
10. 阿奇霉素分散片 (Azithromycin Dispersible Tablets) - 205 occurrences
```

**Observation**: Antibiotics ครอบงำ top 10 (Levofloxacin, Cephalexin, Erythromycin, Ampicillin, Azithromycin, Cefazolin)

## System Architecture

```
Disease Description (desc)
        ↓
   DeepSeek API
        ↓
  LLM Prediction
        ↓
 Drug Name Extraction
        ↓
   Ranked List of Drugs
        ↓
  Evaluation (P@K, R@K)
```

## Components

### 1. DescriptionDrugDataset

Load และ process `medical.json`:

```python
from recommend_from_desc_to_recommand_drug import DescriptionDrugDataset

dataset = DescriptionDrugDataset("../data/medical.json")
print(f"Total: {len(dataset)} samples")

# Get sample
sample = dataset.get_sample(0)
print(f"Disease: {sample['disease_name']}")
print(f"Description: {sample['desc']}")
print(f"Ground Truth: {sample['ground_truth']}")
```

**Sample Structure:**
```python
{
    'disease_name': str,      # ชื่อโรค
    'desc': str,              # คำอธิบายโรค (input)
    'ground_truth': set,      # Set ของยาที่ถูกต้อง (output)
    'entity': dict            # Full entity จาก medical.json
}
```

### 2. DescriptionDrugPredictor

Predict drugs using DeepSeek API:

```python
from recommend_from_desc_to_recommand_drug import DescriptionDrugPredictor

predictor = DescriptionDrugPredictor(model="deepseek-chat")

# Predict
description = "百日咳是由百日咳杆菌所致的急性呼吸道传染病..."
recommendations = predictor.predict(description, top_k=30)

print(f"Top 10: {recommendations[:10]}")
```

**Prompt Engineering:**
```python
prompt = f"""你是一个专业的医学助手。根据以下疾病描述，推荐适合的药物治疗方案。

疾病描述：
{description}

要求：
1. 请直接列出推荐的药物名称，每个药物用顿号（、）分隔
2. 按照推荐优先级从高到低排序
3. 只输出药物名称，不要输出任何解释或说明
4. 尽可能多推荐一些药物（至少15个）
5. 优先推荐常用、有效的药物

推荐药物："""
```

### 3. RecommenderMetrics

Calculate Precision@K and Recall@K:

```python
from recommend_from_desc_to_recommand_drug import RecommenderMetrics

metrics = RecommenderMetrics()

# Evaluate single
recommended = ["药A", "药B", "药C", "药D", "药E"]
relevant = {"药A", "药C", "药F"}

result = metrics.evaluate_single(recommended, relevant)
print(f"P@5: {result[5]['precision']}")
print(f"R@5: {result[5]['recall']}")

# Accumulate
metrics.update(recommended, relevant)

# Get aggregate
agg = metrics.get_aggregate_metrics()
metrics.print_summary("Test")
```

**K Values**: 5, 10, 15, 20, 30

**Why these K values?**
- Average drugs per disease = 7.78
- K=5: Strict precision (ต่ำกว่า average)
- K=10: Slightly above average
- K=15, 20: Double the average
- K=30: High recall coverage

## Usage

### Prerequisites

```bash
# Install dependencies
pip install openai numpy

# Set API key
export DEEPSEEK_API_KEY='your-api-key-here'
```

### Run Evaluation

```bash
cd src
python recommend_from_desc_to_recommand_drug.py
```

**Configuration** (in `main()` function):

```python
config = {
    'data_path': '../data/medical.json',
    'model': 'deepseek-chat',
    'num_samples': 20,          # Test samples
    'max_recommendations': 30   # Max drugs to predict
}
```

### Example Output

```
================================================================================
Sample 1/20
================================================================================

Disease: 百日咳
Description: 百日咳是由百日咳杆菌所致的急性呼吸道传染病。其特征为阵发性痉挛性咳嗽...
Ground Truth (6 drugs): ['红霉素肠溶片', '穿心莲内酯片', '百咳静糖浆', ...]

Top 10 Predictions: ['红霉素肠溶片', '琥乙红霉素片', '阿奇霉素片', ...]
Inference Time: 1.23s

────────────────────────────────────────────────────────────────────────────────
Sample Metrics
────────────────────────────────────────────────────────────────────────────────
@ 5 - P: 0.4000, R: 0.3333
@10 - P: 0.3000, R: 0.5000
@15 - P: 0.2667, R: 0.6667
@20 - P: 0.2000, R: 0.6667
@30 - P: 0.1333, R: 0.6667

================================================================================
Baseline Drug Recommendation Metrics
================================================================================
Total samples: 20

Metric          @5        @10       @15       @20       @30
────────────────────────────────────────────────────────────────────────────────
Precision       0.3800    0.2400    0.1867    0.1550    0.1067
Recall          0.3210    0.4560    0.5340    0.5890    0.6420

================================================================================
Analysis
================================================================================

Optimal K Analysis:
Looking for best trade-off between Precision and Recall...

K= 5: P=0.3800, R=0.3210, F1=0.3479
K=10: P=0.2400, R=0.4560, F1=0.3158
K=15: P=0.1867, R=0.5340, F1=0.2761
K=20: P=0.1550, R=0.5890, F1=0.2452
K=30: P=0.1067, R=0.6420, F1=0.1822

🎯 Optimal K (best F1): 5
```

## Evaluation Metrics

### Precision@K

```
Precision@K = (Number of relevant drugs in top-K) / K
```

**Interpretation**:
- P@5 = 0.4 → 40% ของ top-5 predictions ถูกต้อง
- สูง = accurate predictions
- ต่ำ = many false positives

### Recall@K

```
Recall@K = (Number of relevant drugs in top-K) / (Total relevant drugs)
```

**Interpretation**:
- R@10 = 0.5 → พบ 50% ของยาที่ถูกต้องใน top-10
- สูง = comprehensive coverage
- ต่ำ = missing many correct drugs

### F1 Score

```
F1@K = 2 × (Precision@K × Recall@K) / (Precision@K + Recall@K)
```

**Interpretation**:
- Harmonic mean of P and R
- Balance between accuracy and coverage
- Use to find optimal K

## Optimal K Selection

### Trade-off Analysis

```
K ↑  →  Precision ↓,  Recall ↑
K ↓  →  Precision ↑,  Recall ↓
```

### Finding Optimal K

1. Calculate F1@K for each K
2. Select K with **highest F1 score**
3. Consider practical constraints:
   - Medical context: prefer high precision (avoid wrong drugs)
   - Top-5 or Top-10 is realistic for doctors to review

### Expected Results

Based on dataset statistics (avg 7.78 drugs):

**Expected Optimal K**: 5-10

**Reasoning**:
- K=5: Below average, high precision
- K=10: Slightly above average, balanced
- K>15: Too many predictions, low precision

## Challenges

### 1. Large Drug Space

- **3,800 unique drugs** to choose from
- LLM may predict drugs not in ground truth but still valid
- Exact match evaluation is strict

### 2. Drug Name Variations

Examples:
- 红霉素肠溶片 vs 红霉素片
- 阿奇霉素片 vs 阿奇霉素分散片
- Generic vs brand names

**Current approach**: Exact string matching
**Future**: Fuzzy matching or drug normalization

### 3. LLM Hallucination

- LLM may invent drug names
- LLM may use generic names while ground truth uses specific forms
- No validation against actual drug database

### 4. Limited Context

**Baseline (No KG)**:
- Only disease description
- No symptom details
- No patient demographics
- No contraindications

**Future with KG**:
- Add symptom information
- Add drug interactions
- Add side effects
- Add patient factors

## Limitations

1. **No KG context**: Baseline model only
2. **Exact matching**: Doesn't handle drug name variations
3. **No drug validation**: May predict non-existent drugs
4. **Language barrier**: Chinese medical terminology
5. **No ranking confidence**: All predictions weighted equally

## Next Steps

### 1. Add Knowledge Graph Enhancement

```python
# Enhance with KG context
kg_context = kg.search_relevant_context(disease_name)
recommendations_kg = predictor.predict_with_kg(description, kg_context)
```

### 2. Improve Drug Extraction

- Normalize drug names
- Handle variations (片/胶囊/颗粒)
- Fuzzy matching for evaluation

### 3. Add Drug Frequency Prior

```python
# Bias predictions toward frequent drugs
top_frequent = dataset.get_top_drugs(100)
# Use in prompt or post-processing
```

### 4. Multi-metric Evaluation

Add:
- MAP@K (Mean Average Precision)
- MRR@K (Mean Reciprocal Rank)
- NDCG@K (Normalized Discounted Cumulative Gain)

### 5. Error Analysis

- Which diseases have lowest P@K?
- Which drug categories are hardest to predict?
- Common false positives/negatives?

## Comparison with Other Tasks

### vs. Symptom→Drug Recommendation

| Aspect | Desc→Drug (This) | Symptom→Drug |
|--------|------------------|--------------|
| Input | Long description | Short symptom list |
| Context | Rich medical info | Limited symptoms |
| Difficulty | Moderate | Harder (less context) |

### vs. Disease→Drug (Direct)

| Aspect | Desc→Drug (This) | Disease→Drug |
|--------|------------------|--------------|
| Input | Description text | Disease name |
| Task | NLP-heavy | Lookup-heavy |
| Real-world | More realistic | Too simple |

## File Structure

```
recommend_from_desc_to_recommand_drug.py
├── DescriptionDrugDataset      # Load desc→drug pairs
├── DescriptionDrugPredictor    # Predict with DeepSeek API
├── RecommenderMetrics          # Calculate P@K, R@K
└── main()                      # Run evaluation
```

## Performance Tips

### 1. Batch Processing

```python
# For large evaluations, batch API calls
for batch in batches(dataset.samples, batch_size=10):
    results = predictor.predict_batch(batch)
```

### 2. Caching

```python
# Cache predictions to avoid re-running
import json
with open('predictions_cache.json', 'w') as f:
    json.dump(results, f)
```

### 3. Parallel Execution

```python
# Use multiprocessing for metrics calculation
from multiprocessing import Pool
```

## References

- Dataset: [QASystemOnMedicalKG](https://github.com/zhihao-chen/QASystemOnMedicalKG)
- Model: [DeepSeek API](https://platform.deepseek.com)
- Metrics: Information Retrieval standard metrics

## Citation

```bibtex
@misc{medical_drug_recommendation,
  title={Drug Recommendation from Disease Description},
  author={Your Name},
  year={2025},
  note={Baseline model using DeepSeek API}
}
```

## Support

For issues or questions:
- Check dataset loading: `python -c "from recommend_from_desc_to_recommand_drug import DescriptionDrugDataset; d = DescriptionDrugDataset()"`
- Verify API key: `echo $DEEPSEEK_API_KEY`
- Check Python version: Python 3.8+
