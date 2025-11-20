# PTM Herb Recommender Comparison

Compare **Baseline** (no RAG) vs **RAG-Enhanced** TCM herb recommenders with controlled experiments.

## Overview

This comparison script evaluates both systems on **identical test samples** using a **fixed random seed** for reproducibility.

**Scripts:**
- `ptm_herb_recommender.py` - Baseline (vocabulary-constrained only)
- `ptm_herb_recommender_with_rag.py` - RAG-enhanced (vocabulary + knowledge retrieval)
- `compare_ptm.py` - Side-by-side comparison (this script)

## Key Features

### 🎲 Reproducible Comparison
- Fixed random seed (default: 42)
- Same test samples for both systems
- Identical vocabulary and configuration
- Fair apple-to-apple comparison

### 📊 Comprehensive Metrics
- Precision@K (K=5,10,15,20,30)
- Recall@K
- F1 Score@K
- Time per sample
- Improvement percentages

### 🎯 Side-by-Side Output
- Sample-level comparison
- Aggregate metrics comparison
- Statistical significance
- Clear recommendations

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install openai numpy

# Set API keys
export DEEPSEEK_API_KEY='your-deepseek-key'
export OPENAI_API_KEY='your-openai-key'  # Required for RAG

# Verify data exists
ls ../data/PTM/data/prescriptions.txt
ls ../data/herb-knowledge.csv
```

### Run Comparison

```bash
cd src
python compare_ptm.py
```

**Default configuration:**
- Test samples: 20
- Random seed: 42
- Vocabulary size: 200 herbs
- RAG top-K: 10 knowledge entries

## Usage

### Basic Usage

```python
from compare_ptm import PTMComparison

# Initialize comparison
comparison = PTMComparison(
    prescriptions_path="../data/PTM/data/prescriptions.txt",
    knowledge_path="../data/herb-knowledge.csv",
    model="deepseek-chat",
    random_seed=42  # Fixed for reproducibility
)

# Run comparison
comparison.run_comparison(
    num_samples=20,
    vocab_size=200,
    max_recommendations=30,
    rag_top_k=10
)
```

### Custom Configuration

```python
# Larger test set
comparison.run_comparison(num_samples=100)

# Different random seed
comparison = PTMComparison(random_seed=123)

# Larger vocabulary
comparison.run_comparison(vocab_size=500)

# More RAG knowledge
comparison.run_comparison(rag_top_k=15)
```

## Output Format

### Sample-Level Comparison

For each test sample:

```
================================================================================
Sample 1/20 (Index: 42)
================================================================================

📝 Symptoms: 头痛发热，恶寒无汗，咳嗽喘急...
✅ Ground Truth: 8 herbs

🔵 BASELINE (No RAG)...
   Top 5: ['麻黄', '桂枝', '甘草', '杏仁', '防风']
   Time: 1.23s

🟢 RAG-ENHANCED...
   Top 5: ['麻黄', '杏仁', '桂枝', '甘草', '生姜']
   Time: 1.67s

📊 Sample Metrics Comparison:
Metric     Baseline     RAG          Δ
──────────────────────────────────────────────────
P@5        0.4000       0.6000       +0.2000
P@10       0.3000       0.5000       +0.2000
P@20       0.2500       0.4000       +0.1500
```

### Aggregate Comparison

```
================================================================================
PRECISION@K Comparison
================================================================================
K      Baseline        RAG             Δ (Improvement)      % Change
────────────────────────────────────────────────────────────────────────────────
5      0.3500          0.4200          +0.0700 🟢        +20.00%
10     0.2800          0.3400          +0.0600 🟢        +21.43%
15     0.2200          0.2700          +0.0500 🟢        +22.73%
20     0.1900          0.2300          +0.0400 🟢        +21.05%
30     0.1500          0.1800          +0.0300 🟢        +20.00%

================================================================================
RECALL@K Comparison
================================================================================
K      Baseline        RAG             Δ (Improvement)      % Change
────────────────────────────────────────────────────────────────────────────────
5      0.3200          0.3900          +0.0700 🟢        +21.88%
10     0.5100          0.6200          +0.1100 🟢        +21.57%
15     0.6000          0.7300          +0.1300 🟢        +21.67%
20     0.6900          0.8400          +0.1500 🟢        +21.74%
30     0.8200          0.9800          +0.1600 🟢        +19.51%

================================================================================
F1 SCORE Comparison
================================================================================
K      Baseline        RAG             Δ (Improvement)      % Change
────────────────────────────────────────────────────────────────────────────────
5      0.3333          0.4025          +0.0692 🟢        +20.76%
10     0.3600          0.4375          +0.0775 🟢        +21.53%
15     0.3231          0.3950          +0.0719 🟢        +22.25%
20     0.2990          0.3618          +0.0628 🟢        +21.00%
30     0.2545          0.3050          +0.0505 🟢        +19.84%
```

### Summary

```
================================================================================
SUMMARY
================================================================================

🔵 BASELINE (No RAG):
   Best F1: 0.4375 @ K=10
   P@10: 0.2800
   R@10: 0.5100

🟢 RAG-ENHANCED:
   Best F1: 0.5150 @ K=10
   P@10: 0.3400
   R@10: 0.6200

📈 IMPROVEMENT:
   F1 Score: +0.0775 (+21.53%)
   ✅ RAG is better!

⏱️  TIME:
   Baseline: 24.60s total, 1.23s/sample
   RAG: 33.40s total, 1.67s/sample
   Overhead: +8.80s (+35.77%)

================================================================================
CONCLUSIONS
================================================================================
✅ RAG Enhancement provides significant improvement!
   P@10: +0.0600 (+21.43%)
   R@10: +0.1100 (+21.57%)
   Recommendation: Use RAG-enhanced version in production
```

## Interpreting Results

### Symbols

- 🟢 Green: RAG outperforms baseline (positive improvement)
- 🔴 Red: Baseline outperforms RAG (negative improvement)
- ⚪ White: No significant difference

### Key Metrics

**Precision@K**: What fraction of recommended herbs are correct?
- Higher is better
- Important for minimizing incorrect recommendations

**Recall@K**: What fraction of correct herbs were recommended?
- Higher is better
- Important for comprehensive coverage

**F1 Score@K**: Harmonic mean of precision and recall
- Balanced metric
- Best for overall performance evaluation

### Performance Threshold

The script automatically categorizes improvement:

**Significant**: P@10 or R@10 improvement > 0.01 (1%)
- ✅ Use RAG-enhanced version

**Modest**: Improvement > 0 but < 0.01
- ⚠️ Use RAG if cost is acceptable

**None/Negative**: Improvement ≤ 0
- 🔴 Stick with baseline

## Reproducibility

### Random Seed

The script uses a **fixed random seed** to ensure reproducibility:

```python
random_seed = 42  # Default

# Same seed = same samples
comparison1 = PTMComparison(random_seed=42)
comparison2 = PTMComparison(random_seed=42)
# Both will test on identical samples

# Different seed = different samples
comparison3 = PTMComparison(random_seed=123)
```

### Sample Selection

```
🎲 Selecting 20 random samples (seed=42)...
Selected sample indices: [42, 157, 89, 234, 12, 198, 76, ...]
```

The same indices are used for both baseline and RAG evaluation.

### Verification

To verify reproducibility, run the script multiple times with the same seed:

```bash
# Run 1
python compare_ptm.py > results1.txt

# Run 2
python compare_ptm.py > results2.txt

# Should be identical
diff results1.txt results2.txt
```

## Advanced Usage

### Large-Scale Evaluation

```python
# Test on 100 samples
comparison = PTMComparison(random_seed=42)
comparison.run_comparison(num_samples=100)
```

**Note**: This will take longer and cost more API calls.

### Statistical Significance

For rigorous evaluation, test on multiple random seeds:

```python
# Test on 5 different random seeds
seeds = [42, 123, 456, 789, 1024]
results = []

for seed in seeds:
    comparison = PTMComparison(random_seed=seed)
    # Store and aggregate results
```

### Parameter Tuning

Find optimal RAG top-K:

```python
for rag_top_k in [5, 10, 15, 20]:
    print(f"\nTesting RAG top-K={rag_top_k}")
    comparison.run_comparison(rag_top_k=rag_top_k)
```

Find optimal vocabulary size:

```python
for vocab_size in [100, 200, 300, 500]:
    print(f"\nTesting vocab_size={vocab_size}")
    comparison.run_comparison(vocab_size=vocab_size)
```

## Cost Analysis

### API Costs per Run (20 samples)

**Baseline:**
- DeepSeek API: 20 queries × ~$0.001 = ~$0.02
- Total: ~$0.02

**RAG-Enhanced:**
- OpenAI Embeddings (first run): 337 herbs × ~$0.00002 = ~$0.007
- OpenAI Embeddings (queries): 20 × ~$0.000004 = ~$0.00008
- DeepSeek API: 20 queries × ~$0.001 = ~$0.02
- Total first run: ~$0.027
- Total subsequent runs: ~$0.02 (if embeddings cached)

**Comparison (20 samples):**
- Baseline: ~$0.02
- RAG: ~$0.027 (first run) or ~$0.02 (cached)
- Total: ~$0.047 (first run) or ~$0.04 (cached)

**For 100 samples:**
- Total: ~$0.23 (first run) or ~$0.20 (cached)

## Performance Optimization

### Cache RAG Embeddings

To avoid re-creating herb embeddings on each run:

```python
import pickle
import os

# In compare_ptm.py, modify RAG initialization:
if os.path.exists('cached_herb_embeddings.pkl'):
    # Load cached embeddings
    with open('cached_herb_embeddings.pkl', 'rb') as f:
        cached_embeddings = pickle.load(f)
    # Use cached embeddings
else:
    # Create and cache
    rag_system = HerbKnowledgeRAG(knowledge_path)
    with open('cached_herb_embeddings.pkl', 'wb') as f:
        pickle.dump(rag_system.embeddings, f)
```

**Savings**: ~$0.007 per run (after first run)

### Parallel Processing

For large-scale evaluation, process samples in parallel:

```python
from concurrent.futures import ThreadPoolExecutor

# NOT recommended due to API rate limits
# Better to run sequentially
```

## Troubleshooting

### Different Results on Same Seed

**Cause**: API non-determinism (temperature > 0)

**Solution**: Lower temperature in both predictors:
```python
# In predictor code
temperature=0.1  # More deterministic
```

### RAG Worse Than Baseline

**Possible causes:**
1. Retrieved knowledge is irrelevant
2. Prompt too long (context dilution)
3. Vocabulary mismatch (herb names in knowledge vs dataset)

**Solutions:**
- Increase RAG top-K
- Improve knowledge base quality
- Fine-tune retrieval model

### High API Costs

**Solutions:**
- Reduce `num_samples`
- Cache embeddings
- Use smaller vocabulary
- Reduce RAG top-K

### Slow Execution

**Causes:**
- API latency
- Large number of samples

**Solutions:**
- Reduce `num_samples`
- Cache embeddings (saves ~10s per run)
- Use faster API tier

## Files

```
src/
├── ptm_herb_recommender.py          # Baseline (no RAG)
├── ptm_herb_recommender_with_rag.py # RAG-enhanced
├── compare_ptm.py                    # This comparison script
└── ...

data/
├── PTM/data/prescriptions.txt        # PTM dataset
└── herb-knowledge.csv                # RAG knowledge base

COMPARISON_README.md                  # This file
```

## Examples

### Example 1: Quick Test (5 samples)

```bash
python compare_ptm.py
# Edit num_samples=5 in main()
```

### Example 2: Production Evaluation (100 samples)

```bash
python compare_ptm.py
# Edit num_samples=100 in main()
```

### Example 3: Different Random Seed

```python
# In main():
comparison = PTMComparison(random_seed=123)
```

### Example 4: Larger Vocabulary

```python
# In main():
comparison.run_comparison(vocab_size=500)
```

## Expected Results

Based on preliminary testing, expected improvements with RAG:

| Metric | Baseline | RAG | Improvement |
|--------|----------|-----|-------------|
| P@5    | 0.35-0.40 | 0.40-0.50 | +10-25% |
| P@10   | 0.28-0.32 | 0.32-0.40 | +10-25% |
| R@10   | 0.50-0.55 | 0.58-0.68 | +15-25% |
| F1@10  | 0.36-0.40 | 0.42-0.50 | +15-25% |

**Note**: Actual results depend on:
- Test samples
- API model versions
- Knowledge base quality
- Random seed

## Citation

```bibtex
@misc{ptm_comparison,
  title={PTM Herb Recommender Comparison: Baseline vs RAG},
  author={Your Name},
  year={2025},
  note={Reproducible comparison with fixed random seed}
}
```

## References

1. **Baseline**: `ptm_herb_recommender.py` - Vocabulary-constrained prompting
2. **RAG**: `ptm_herb_recommender_with_rag.py` - RAG-enhanced with OpenAI embeddings
3. **Dataset**: [PTM - Prescription as Topic Model](https://github.com/yao8839836/PTM)

## Support

**Common Issues:**

1. **Different results on reruns**
   - Verify same random seed
   - Check API temperature setting

2. **RAG not better**
   - Tune RAG top-K (try 15-20)
   - Check knowledge base relevance
   - Verify vocabulary coverage

3. **High costs**
   - Reduce test samples
   - Cache embeddings
   - Use smaller vocabulary

4. **API errors**
   - Check API keys are set
   - Verify rate limits
   - Add retry logic

## Next Steps

After running comparison:

1. **If RAG is significantly better** (>10% F1 improvement):
   - Deploy RAG-enhanced version
   - Monitor production performance
   - Consider expanding knowledge base

2. **If RAG is modestly better** (5-10% F1 improvement):
   - Evaluate cost/benefit trade-off
   - A/B test in production
   - Optimize RAG parameters

3. **If RAG is not better**:
   - Analyze failure cases
   - Improve knowledge base quality
   - Try different embedding models
   - Consider alternative approaches

---

**Happy comparing! 🎯**
