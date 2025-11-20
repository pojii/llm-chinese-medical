# PTM Herb Recommender - Symptom to TCM Herbs

## Overview

Traditional Chinese Medicine (TCM) herb recommendation system based on symptom descriptions.

**Task**: Symptom → TCM Herbs
**Dataset**: PTM (Prescription as Topic Model) - 98,281 prescriptions
**Baseline**: LLM without Knowledge Graph
**Key Feature**: **Constrained vocabulary** to prevent hallucination

## Task Definition

```
Input:  Symptom description (症状描述)
Output: Ranked list of TCM herbs (中药列表)

Example:
Input:  "头痛发热咳嗽" (Headache, fever, cough)
Output: [麻黄, 桂枝, 杏仁, 甘草, 生姜, 防风, ...]
```

## Dataset Statistics

```
Source: PTM (https://github.com/yao8839836/PTM)
Total prescriptions: 98,281
Unique herbs: ~1,000-2,000 (after cleaning)
Average herbs per prescription: 6.17

Top 3 herbs:
1. 甘草 (Licorice) - 17,544 times (2.9%)
2. 当归 (Angelica) - 11,404 times (1.9%)
3. 人参 (Ginseng) - 11,080 times (1.8%)
```

## Vocabulary Coverage

Using constrained vocabulary to prevent LLM hallucination:

| Vocabulary Size | Coverage | Use Case |
|----------------|----------|----------|
| Top 50 | 32.20% | Strict, common herbs only |
| Top 100 | 43.04% | Balanced |
| **Top 200** | **53.49%** | **Recommended** ✅ |
| Top 500 | 64.99% | Comprehensive |

**Recommendation**: Use **Top 200 herbs** (53% coverage, manageable prompt size)

## Key Innovation: Constrained Vocabulary

### Problem with Unconstrained LLM

```python
# Without vocabulary constraint
LLM may hallucinate herb names:
❌ "板蓝根冲剂" (invented name)
❌ "清热解毒丸" (generic description, not specific herb)
❌ "感冒灵颗粒" (modern medicine, not TCM herb)
```

### Solution: Vocabulary-Constrained Prompting

```python
prompt = f"""
【可用中药列表】（只能从这个列表中选择）
甘草、当归、人参、白术、防风、黄芩、... (Top 200 herbs)

【要求】
1. 只能推荐列表中的中药，不要推荐列表外的药物
2. 按优先级从高到低排序
...
"""
```

**Benefits**:
✅ Prevents hallucination
✅ Ensures all predictions are valid herbs
✅ Maintains evaluation accuracy
✅ Forces LLM to rank from known herbs

## System Architecture

```
Symptom Description
        ↓
   DeepSeek API
   (with herb vocabulary constraint)
        ↓
  Herb Name Extraction
        ↓
  Vocabulary Filtering
        ↓
   Ranked List of Herbs
        ↓
  Evaluation (P@K, R@K)
```

## Components

### 1. PTMHerbDataset

Loads and processes PTM prescriptions:

```python
from ptm_herb_recommender import PTMHerbDataset

# Load dataset
dataset = PTMHerbDataset("data/PTM/data/prescriptions.txt")
print(f"Total: {len(dataset)} samples")

# Get sample
sample = dataset.get_sample(0)
print(f"Symptoms: {sample['symptoms']}")
print(f"Ground Truth: {sample['ground_truth']}")

# Get top herbs for vocabulary
top_200_herbs = dataset.get_top_herbs(200)
```

**Sample Structure:**
```python
{
    'symptoms': str,        # Symptom description
    'ground_truth': set,    # Set of herb names
    'herbs_text': str       # Original herb text with dosages
}
```

### 2. PTMHerbPredictor

Predict herbs with vocabulary constraint:

```python
from ptm_herb_recommender import PTMHerbPredictor

# Initialize with vocabulary
herb_vocab = dataset.get_top_herbs(200)
predictor = PTMHerbPredictor(
    herb_vocabulary=herb_vocab,
    model="deepseek-chat"
)

# Predict
symptoms = "头痛发热咳嗽"
recommendations = predictor.predict(symptoms, top_k=30)

print(f"Top 10: {recommendations[:10]}")
```

**Prompt Template:**
```
你是一个专业的中医助手。根据患者的症状描述，从给定的中药列表中推荐适合的中药。

【患者症状】
{symptoms}

【可用中药列表】（只能从这个列表中选择）
甘草、当归、人参、白术、防风、黄芩、木香、川芎、茯苓、黄连、...

【要求】
1. 只能推荐列表中的中药，不要推荐列表外的药物
2. 按优先级从高到低排序
3. 每个中药用顿号（、）分隔
4. 只输出中药名称，不要任何解释
5. 尽量推荐10-20种中药

【推荐中药】：
```

### 3. HerbRecommenderMetrics

Evaluate with Precision@K and Recall@K:

```python
from ptm_herb_recommender import HerbRecommenderMetrics

metrics = HerbRecommenderMetrics()

# Evaluate single
recommended = ["甘草", "当归", "人参", "白术", "防风"]
relevant = {"甘草", "人参", "黄芩", "川芎"}

result = metrics.evaluate_single(recommended, relevant)
print(f"P@5: {result[5]['precision']}")  # 2/5 = 0.40
print(f"R@5: {result[5]['recall']}")      # 2/4 = 0.50

# Accumulate
metrics.update(recommended, relevant)

# Get aggregate
agg = metrics.get_aggregate_metrics()
metrics.print_summary("Baseline")
```

**K Values**: 5, 10, 15, 20, 30

## Usage

### Prerequisites

```bash
# Install dependencies
pip install openai numpy

# Set API key
export DEEPSEEK_API_KEY='your-api-key-here'

# Ensure PTM dataset is downloaded
ls data/PTM/data/prescriptions.txt
```

### Run Evaluation

```bash
cd src
python ptm_herb_recommender.py
```

**Configuration** (in `main()` function):

```python
config = {
    'prescriptions_path': 'data/PTM/data/prescriptions.txt',
    'model': 'deepseek-chat',
    'num_samples': 20,          # Test samples
    'vocab_size': 200,          # Herb vocabulary size
    'max_recommendations': 30   # Max herbs to predict
}
```

### Expected Output

```
================================================================================
Sample 1/20
================================================================================

Symptoms: 头痛发热，恶寒无汗，咳嗽喘急
Ground Truth (8 herbs): ['麻黄', '桂枝', '杏仁', '甘草', '生姜', ...]

Top 10 Predictions: ['麻黄', '桂枝', '甘草', '杏仁', '防风', ...]
Inference Time: 1.45s

────────────────────────────────────────────────────────────────────────────────
Sample Metrics
────────────────────────────────────────────────────────────────────────────────
@ 5 - P: 0.6000, R: 0.3750
@10 - P: 0.5000, R: 0.6250
@15 - P: 0.4000, R: 0.7500
@20 - P: 0.3500, R: 0.8750
@30 - P: 0.2667, R: 1.0000

================================================================================
Baseline TCM Herb Recommendation Metrics
================================================================================
Total samples: 20

Metric          @5        @10       @15       @20       @30
────────────────────────────────────────────────────────────────────────────────
Precision       0.4200    0.3100    0.2533    0.2150    0.1733
Recall          0.3850    0.5240    0.6180    0.6890    0.7620

🎯 Optimal K (best F1): 10
```

## Evaluation Metrics

### Precision@K

```
Precision@K = (Relevant herbs in top-K) / K
```

**Interpretation**:
- P@5 = 0.42 → 42% of top-5 predictions are correct
- Higher = more accurate predictions

### Recall@K

```
Recall@K = (Relevant herbs in top-K) / (Total relevant herbs)
```

**Interpretation**:
- R@10 = 0.52 → Found 52% of correct herbs in top-10
- Higher = better coverage

### F1 Score

```
F1@K = 2 × (Precision@K × Recall@K) / (Precision@K + Recall@K)
```

Used to find optimal K value.

## Optimal K Selection

### Expected Optimal K

Based on average 6.17 herbs per prescription:

```
K = 5   → Below average (high precision)
K = 10  → ~1.6x average (balanced) ← Likely optimal
K = 15  → ~2.4x average
K = 20  → ~3.2x average
K = 30  → ~4.9x average (high recall)
```

**Expected optimal**: K=10 or K=15

## Advantages

### 1. Constrained Vocabulary ✅

**Problem**: LLM may hallucinate non-existent herbs
**Solution**: Provide explicit herb list in prompt
**Result**: All predictions are valid herbs from dataset

### 2. Frequency-Based Vocabulary ✅

**Top 200 herbs cover 53.49%** of all herb usage:
- Focuses on commonly used herbs
- Reduces prompt size
- Improves LLM focus

### 3. Clean Evaluation ✅

Since all predictions come from vocabulary:
- No need for fuzzy matching
- Exact string comparison works
- Reliable metrics

## Limitations

### 1. Vocabulary Coverage

**53% coverage** means:
- 47% of herb usages are outside vocabulary
- May miss rare/specialized herbs
- Trade-off: larger vocab = longer prompt

**Solution**:
- Increase vocab size to 500 (65% coverage)
- Or use adaptive vocabulary per symptom category

### 2. No Herb Properties

Baseline doesn't consider:
- Herb temperature (warm/cool)
- Toxicity
- Contraindications
- Herb-herb interactions

**Future**: Add herb property knowledge

### 3. No Dosage Information

Only predicts herb names, not:
- Dosage amounts
- Preparation methods
- Administration routes

### 4. Context Independence

Each prediction is independent:
- No patient history
- No syndrome differentiation (辨证论治)
- No consideration of constitution (体质)

## Comparison with Medical.json Dataset

| Aspect | PTM (TCM Herbs) | Medical.json (Modern Drugs) |
|--------|-----------------|----------------------------|
| Domain | Traditional Chinese Medicine | Modern Western Medicine |
| Items | ~1,000-2,000 herbs | 3,800 drugs |
| Avg per case | 6.17 herbs | 7.78 drugs |
| Input | Symptom description | Disease description |
| Coverage (Top 200) | 53.49% | N/A |
| Vocabulary constraint | Yes (prevents hallucination) | Not used |

## Next Steps

### 1. Add Knowledge Graph

```python
# Enhance with herb properties
herb_properties = {
    '甘草': {'temp': 'neutral', 'toxic': 'no', 'category': 'tonic'},
    '麻黄': {'temp': 'warm', 'toxic': 'low', 'category': 'releasing'},
    ...
}
```

### 2. Expand Vocabulary Dynamically

```python
# Adjust vocab size based on symptom category
if '发热' in symptoms:
    vocab = heat_clearing_herbs
elif '虚证' in symptoms:
    vocab = tonic_herbs
```

### 3. Add Syndrome Differentiation

```python
# Identify TCM syndrome first
syndrome = identify_syndrome(symptoms)  # e.g., "风寒感冒"
herbs = recommend_for_syndrome(syndrome)
```

### 4. Multi-Stage Prediction

```python
# Stage 1: Predict herb categories
categories = predict_categories(symptoms)  # ['清热药', '补气药']

# Stage 2: Predict specific herbs from categories
herbs = predict_herbs_from_categories(symptoms, categories)
```

## Files

```
src/ptm_herb_recommender.py
├── PTMHerbDataset          # Load PTM prescriptions
├── PTMHerbPredictor        # Predict with vocab constraint
├── HerbRecommenderMetrics  # Calculate P@K, R@K
└── main()                  # Run evaluation
```

## Performance Tips

### 1. Vocabulary Size Tuning

```python
# Test different vocabulary sizes
for vocab_size in [100, 200, 300, 500]:
    herbs = dataset.get_top_herbs(vocab_size)
    coverage = calculate_coverage(herbs, dataset)
    print(f"Size {vocab_size}: {coverage:.2f}% coverage")
```

### 2. Batch Processing

```python
# Process multiple samples in parallel
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=5) as executor:
    results = executor.map(predictor.predict, symptoms_list)
```

### 3. Response Caching

```python
# Cache predictions to avoid re-running API
import json
cache = {}
if symptoms in cache:
    return cache[symptoms]
else:
    result = predictor.predict(symptoms)
    cache[symptoms] = result
    return result
```

## References

1. **Dataset**: [PTM - Prescription as Topic Model](https://github.com/yao8839836/PTM)
2. **Paper**: Topic Modeling for TCM Prescriptions
3. **API**: [DeepSeek Platform](https://platform.deepseek.com)

## Citation

```bibtex
@misc{ptm_herb_recommender,
  title={TCM Herb Recommendation from Symptoms},
  author={Your Name},
  year={2025},
  note={Baseline model with vocabulary constraint}
}
```

## Disclaimer

⚠️ **Medical Disclaimer**: This system is for **research and educational purposes only**. TCM herb recommendations should NOT be used for actual medical treatment. Always consult qualified TCM practitioners.

⚠️ **Dataset License**: PTM dataset is for research use only (non-commercial).

## Support

**Common Issues**:

1. **API Key Error**
   ```bash
   export DEEPSEEK_API_KEY='your-key'
   ```

2. **Dataset Not Found**
   ```bash
   git clone https://github.com/yao8839836/PTM.git data/PTM
   ```

3. **Import Error**
   ```bash
   pip install openai numpy
   ```

4. **Low Coverage**
   ```python
   # Increase vocabulary size
   config['vocab_size'] = 500  # Up to 65% coverage
   ```
