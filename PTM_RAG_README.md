# PTM Herb Recommender with RAG Enhancement

## Overview

RAG-enhanced Traditional Chinese Medicine (TCM) herb recommendation system that retrieves relevant herb knowledge to improve prediction quality.

**Enhancement**: Retrieval-Augmented Generation (RAG) using `herb-knowledge.csv`

**Baseline**: `ptm_herb_recommender.py` (without RAG)

**This Version**: `ptm_herb_recommender_with_rag.py` (with RAG)

## Key Innovation: RAG + Vocabulary Constraint

This system combines two powerful techniques:

1. **Vocabulary-Constrained Prompting**: Prevents LLM hallucination by providing explicit herb list
2. **RAG Knowledge Retrieval**: Enhances predictions with relevant herb properties and indications

```
Symptom Query
     ↓
Vector Retrieval → Top 10 Herb Knowledge
     ↓                    ↓
  Symptom + Retrieved Knowledge + Herb Vocabulary
                ↓
           DeepSeek API
                ↓
        Ranked Herb List
```

## RAG System Architecture

### 1. Knowledge Base: `herb-knowledge.csv`

**Source**: 337 herb knowledge entries

**Fields**:
- `Pinyin Name`: Chinese pinyin (e.g., MA HUANG, GAN CAO)
- `English Name`: English translation
- `Latin Name`: Scientific botanical name
- `Attributes`: Temperature and taste (e.g., "Warm, Pungent")
- `Meridians/Energy_channels`: Affected meridians
- `Use Part`: Plant part used (root, leaf, etc.)
- `Effect`: Medicinal effects
- `Indication`: Symptoms and conditions treated

**Example Entry**:
```
Pinyin: MA HUANG
English: Chinese Ephedra
Attributes: Warm, Pungent, Slightly Bitter
Effect: To effuse sweat and resolve exterior, diffuse lung and calm asthma
Indication: Wind-cold exterior repletion syndrome, headache, cough, fever, asthma
```

### 2. Vector Embeddings

**Model**: OpenAI `text-embedding-3-small` (via API)
- No local model download required
- Supports Chinese and English
- 1536-dimensional embeddings
- Fast API-based inference
- Cost: ~$0.02 per 1M tokens

**Embedding Text** (per herb):
```
Properties: {Attributes}. Effects: {Effect}. Treats: {Indication}
```

**Example**:
```
Properties: Warm, Pungent, Slightly Bitter.
Effects: To effuse sweat and resolve exterior, diffuse lung and calm asthma.
Treats: Wind-cold exterior repletion syndrome, headache, cough, fever, asthma
```

### 3. Retrieval Process

**Input**: Symptom description (e.g., "头痛发热咳嗽")

**Process**:
1. Encode symptom query into vector
2. Calculate cosine similarity with all 337 herb embeddings
3. Rank by similarity score
4. Retrieve top 10 most relevant herbs

**Output**: 10 herb knowledge entries with similarity scores

### 4. Knowledge-Augmented Prompt

```python
prompt = f"""
【参考中药知识】（根据症状检索的相关中药知识）
1. MA HUANG：性味：Warm, Pungent；功效：To diffuse lung and calm asthma；主治：cough, fever, asthma
2. GAN CAO：性味：Mild, Sweet；功效：To harmonize other herbs；主治：Various conditions
...
10. XING REN：性味：Warm, Bitter；功效：To relieve cough；主治：cough and asthma

【患者症状】
{symptoms}

【可用中药列表】（只能从这个列表中选择）
甘草、当归、人参、白术、防风、黄芩、... (Top 200 herbs)

【要求】
1. 参考上述中药知识，选择最适合的中药
2. 只能推荐列表中的中药，不要推荐列表外的药物
3. 按优先级从高到低排序
...

【推荐中药】：
"""
```

## Components

### 1. HerbKnowledgeRAG

RAG system for herb knowledge retrieval.

```python
from ptm_herb_recommender_with_rag import HerbKnowledgeRAG

# Initialize with OpenAI API (no local model)
rag = HerbKnowledgeRAG("../data/herb-knowledge.csv")
# Will use OPENAI_API_KEY from environment

# Retrieve relevant knowledge
symptoms = "头痛发热咳嗽"
knowledge = rag.retrieve_relevant_knowledge(symptoms, top_k=10)

# Each entry contains:
# - pinyin, english, attributes, meridians
# - effect, indication, knowledge_text
# - similarity (cosine similarity score from OpenAI embeddings)
```

**Methods**:
- `load_knowledge()`: Load CSV and parse herb entries
- `initialize_embeddings()`: Create embeddings via OpenAI API (batched)
- `retrieve_relevant_knowledge(query, top_k)`: Retrieve top K relevant entries using OpenAI embeddings

### 2. PTMHerbPredictorWithRAG

Predictor with RAG enhancement.

```python
from ptm_herb_recommender_with_rag import PTMHerbPredictorWithRAG

# Initialize
predictor = PTMHerbPredictorWithRAG(
    herb_vocabulary=top_200_herbs,
    rag_system=rag,
    model="deepseek-chat"
)

# Predict
symptoms = "头痛发热咳嗽"
herbs = predictor.predict(symptoms, top_k=30)

print(f"Top 10: {herbs[:10]}")
```

**Workflow**:
1. Retrieve top 10 relevant herb knowledge
2. Format knowledge into prompt
3. Add herb vocabulary constraint
4. Call DeepSeek API
5. Extract and filter predictions
6. Return ranked herb list

### 3. PTMHerbDataset

Same as baseline (see `PTM_RECOMMENDER_README.md`).

### 4. HerbRecommenderMetrics

Same as baseline (see `PTM_RECOMMENDER_README.md`).

## Usage

### Prerequisites

```bash
# Install dependencies (NO local models required!)
pip install openai numpy

# Set API keys
export DEEPSEEK_API_KEY='your-deepseek-api-key'
export OPENAI_API_KEY='your-openai-api-key'

# Ensure data files exist
ls ../data/PTM/data/prescriptions.txt
ls ../data/herb-knowledge.csv
```

**Required API Keys:**
- **DEEPSEEK_API_KEY**: For LLM inference (herb recommendation)
- **OPENAI_API_KEY**: For embeddings (knowledge retrieval)

### Run Evaluation

```bash
cd src
python ptm_herb_recommender_with_rag.py
```

**Configuration** (in `main()` function):

```python
config = {
    'prescriptions_path': '../data/PTM/data/prescriptions.txt',
    'knowledge_path': '../data/herb-knowledge.csv',
    'model': 'deepseek-chat',
    'num_samples': 20,          # Test samples
    'vocab_size': 200,          # Herb vocabulary size
    'max_recommendations': 30,  # Max herbs to predict
    'rag_top_k': 10            # Knowledge entries to retrieve
}
```

### Expected Output

```
================================================================================
Initializing RAG System with OpenAI Embeddings API
================================================================================
Loading herb knowledge from: ../data/herb-knowledge.csv
Loaded 337 herb knowledge entries
Creating embeddings using OpenAI API...
Processing 337 herb knowledge entries...
  Processing batch 1/4...
  Processing batch 2/4...
  Processing batch 3/4...
  Processing batch 4/4...
✅ Created 337 embeddings using OpenAI API

================================================================================
Sample 1/20
================================================================================

Symptoms: 头痛发热，恶寒无汗，咳嗽喘急
Ground Truth (8 herbs): ['麻黄', '桂枝', '杏仁', '甘草', '生姜', ...]

Retrieved Knowledge (Top 5):
  1. MA HUANG (Chinese Ephedra) - Similarity: 0.723
     Effect: To effuse sweat and resolve exterior, diffuse lung and calm asthma...
  2. XING REN (Apricot Kernel) - Similarity: 0.681
     Effect: To relieve cough and dispel phlegm...
  3. ZI WAN (Tatarion Aster) - Similarity: 0.645
     Effect: To moisten lung and precipitate qi, relieve cough...
  4. GAN JIANG (Dried Ginger) - Similarity: 0.612
     Effect: To warm center and dissipate cold...
  5. GAN CAO (Licorice) - Similarity: 0.598
     Effect: To harmonize other herbs...

Top 10 Predictions: ['麻黄', '桂枝', '甘草', '杏仁', '防风', ...]
Inference Time: 1.67s

────────────────────────────────────────────────────────────────────────────────
Sample Metrics
────────────────────────────────────────────────────────────────────────────────
@ 5 - P: 0.6000, R: 0.3750
@10 - P: 0.5000, R: 0.6250
@15 - P: 0.4000, R: 0.7500
@20 - P: 0.3500, R: 0.8750
@30 - P: 0.2667, R: 1.0000

================================================================================
RAG-Enhanced TCM Herb Recommendation Metrics
================================================================================
Total samples: 20

Metric          @5        @10       @15       @20       @30
────────────────────────────────────────────────────────────────────────────────
Precision       0.4500    0.3300    0.2667    0.2250    0.1800
Recall          0.4100    0.5500    0.6400    0.7100    0.7800

🎯 Optimal K (best F1): 10
```

## RAG Enhancement Benefits

### 1. Contextual Knowledge Injection ✅

**Problem**: LLM has general medical knowledge but may lack specific TCM herb properties

**Solution**: Retrieve and inject relevant herb knowledge:
- Herb temperature (warm/cool)
- Herb taste (pungent/sweet/bitter)
- Medicinal effects
- Indications for specific symptoms

**Example**:
```
Symptom: "发热咳嗽" (fever and cough)
Retrieved: MA HUANG (warm, pungent, treats fever, cough)
Result: LLM prioritizes MA HUANG in recommendations
```

### 2. Improved Herb Selection ✅

**Without RAG**: LLM guesses based on general knowledge

**With RAG**: LLM sees specific herb properties and indications

**Example**:
```
Symptom: "虚寒腹痛" (cold pain from deficiency)

Without RAG:
- May recommend random herbs from vocabulary
- No clear reasoning

With RAG:
- Sees: GAN JIANG (warm, treats cold pain)
- Sees: FU ZI (extremely hot, treats yang deficiency)
- Recommends appropriate warming herbs
```

### 3. Semantic Similarity Matching ✅

**Embedding Model Benefits**:
- Matches Chinese symptoms to Chinese/English herb knowledge
- Handles synonym matching (e.g., "咳嗽" ↔ "cough")
- Captures semantic similarity (e.g., "发热" ↔ "fever and aversion to cold")

### 4. Maintains Vocabulary Constraint ✅

**Important**: RAG provides knowledge, but vocabulary constraint prevents hallucination

**Flow**:
1. RAG retrieves: "MA HUANG is good for fever"
2. LLM wants to recommend "MA HUANG"
3. Checks vocabulary: "麻黄" (MA HUANG in Chinese) is in list ✅
4. Recommends "麻黄"

**If herb not in vocabulary**:
1. RAG retrieves: "RARE_HERB is good for symptom"
2. LLM wants to recommend it
3. Checks vocabulary: NOT in list ❌
4. Skips or chooses similar herb from vocabulary

## Comparison: Baseline vs RAG

### Expected Performance Improvement

| Metric | Baseline (No RAG) | RAG-Enhanced | Expected Δ |
|--------|-------------------|--------------|------------|
| P@5    | ~0.40             | ~0.45        | +12.5%     |
| P@10   | ~0.30             | ~0.33        | +10%       |
| R@10   | ~0.50             | ~0.55        | +10%       |
| F1@10  | ~0.37             | ~0.41        | +10.8%     |

**Note**: Run both systems on same test set to measure actual improvement.

### Why RAG Helps

1. **Better Context**: LLM sees actual herb properties instead of guessing
2. **Focused Selection**: Retrieved knowledge acts as "hints" for relevant herbs
3. **Explainability**: Can trace why certain herbs were recommended
4. **Accuracy**: Reduces random guessing, improves precision

### When RAG May Not Help

1. **Rare Symptoms**: If symptom has no close match in knowledge base
2. **Novel Combinations**: TCM often uses herb combinations not in knowledge base
3. **Retrieval Errors**: If similarity matching fails to find relevant herbs

## Tuning Parameters

### 1. RAG Top-K

```python
# Retrieve more knowledge entries
rag.retrieve_relevant_knowledge(symptoms, top_k=15)  # Default: 10
```

**Trade-off**:
- ↑ More knowledge = More context but longer prompt
- ↓ Less knowledge = Faster but may miss relevant herbs

**Recommended**: 10-15 entries

### 2. Vocabulary Size

```python
herb_vocabulary = dataset.get_top_herbs(300)  # Default: 200
```

**Trade-off**:
- Top 200 = 53.49% coverage, shorter prompt
- Top 500 = 64.99% coverage, longer prompt

**Recommended**: 200-300 herbs

### 3. Embedding Model

```python
# Current: paraphrase-multilingual-MiniLM-L12-v2
# Alternatives:
model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
```

**Trade-off**:
- Smaller models: Faster, less accurate
- Larger models: Slower, more accurate

### 4. Temperature

```python
response = self.client.chat.completions.create(
    temperature=0.3,  # Lower = more deterministic
    top_p=0.85
)
```

**Recommended**: 0.2-0.4 for medical tasks (need consistency)

## Advanced Features

### 1. Similarity Score Filtering

```python
# Only use high-confidence knowledge
knowledge = [k for k in knowledge if k['similarity'] > 0.5]
```

### 2. Weighted Knowledge

```python
# Weight knowledge by similarity
for entry in knowledge:
    weight = entry['similarity']
    # Use weight in prompt formatting
```

### 3. Herb Category Filtering

```python
# Filter by herb category before embedding
if '发热' in symptoms:
    # Only retrieve "heat-clearing" herbs
    filtered_knowledge = [k for k in knowledge if 'clear heat' in k['effect']]
```

### 4. Multi-stage RAG

```python
# Stage 1: Retrieve herb categories
categories = rag.retrieve_categories(symptoms)

# Stage 2: Retrieve specific herbs from those categories
herbs = rag.retrieve_herbs(symptoms, categories)
```

## Performance Tips

### 1. Cache Embeddings (Recommended)

Since embeddings cost money via API, cache them after first run:

```python
import pickle
import os

# Save embeddings after initialization
if not os.path.exists('cached_herb_embeddings.pkl'):
    rag = HerbKnowledgeRAG("../data/herb-knowledge.csv")
    with open('cached_herb_embeddings.pkl', 'wb') as f:
        pickle.dump(rag.embeddings, f)
else:
    # Load from cache (saves API calls)
    rag = HerbKnowledgeRAG.__new__(HerbKnowledgeRAG)
    rag.load_knowledge()
    with open('cached_herb_embeddings.pkl', 'rb') as f:
        rag.embeddings = pickle.load(f)
```

**Cost savings**:
- First run: ~$0.02 for 337 herbs
- Subsequent runs: $0 (use cached embeddings)
- Only query embeddings cost money (~$0.000004 per query)

### 2. Batch Processing

Process multiple samples to optimize API usage:

```python
# Batch queries if possible (but query embeddings are already fast)
# Main cost is in initial herb knowledge embeddings
```

### 3. API Rate Limits

OpenAI API has rate limits:
- Free tier: 3 RPM (requests per minute)
- Paid tier: 3,500 RPM

Current implementation uses batching (100 herbs per request) to minimize API calls.

## Limitations

### 1. Knowledge Coverage

- **337 herbs** in knowledge base
- **~1,000-2,000 herbs** in PTM dataset
- Coverage: ~17-34% of herbs have detailed knowledge

**Solution**: Expand knowledge base or use hierarchical retrieval

### 2. Language Mismatch

- Knowledge base: English + Pinyin
- Dataset: Simplified Chinese
- Need mapping: 麻黄 ↔ MA HUANG

**Current**: Relies on semantic embedding to bridge gap

### 3. Retrieval Quality

- Depends on embedding model quality
- May retrieve irrelevant herbs if symptom description is vague
- No guarantee all relevant herbs are retrieved

### 4. Prompt Length

- Top 10 knowledge entries + 200 herb vocabulary = long prompt
- May hit token limits with very detailed knowledge

**Solution**: Summarize knowledge entries or reduce top-K

## Files

```
src/ptm_herb_recommender_with_rag.py
├── HerbKnowledgeRAG          # RAG system for knowledge retrieval
├── PTMHerbDataset             # Load PTM prescriptions (same as baseline)
├── PTMHerbPredictorWithRAG    # Predict with RAG + vocab constraint
├── HerbRecommenderMetrics     # Calculate P@K, R@K (same as baseline)
└── main()                     # Run evaluation

data/herb-knowledge.csv        # 337 herb knowledge entries

PTM_RAG_README.md             # This file
PTM_RECOMMENDER_README.md     # Baseline documentation
```

## Evaluation Script

### Compare Baseline vs RAG

```python
# Run baseline
baseline_results = run_evaluation(
    predictor_type='baseline',
    num_samples=100
)

# Run RAG
rag_results = run_evaluation(
    predictor_type='rag',
    num_samples=100
)

# Compare
improvement = {
    'precision@10': rag_results['p10'] - baseline_results['p10'],
    'recall@10': rag_results['r10'] - baseline_results['r10']
}
print(f"Precision@10 improvement: {improvement['precision@10']:.4f}")
print(f"Recall@10 improvement: {improvement['recall@10']:.4f}")
```

## Next Steps

### 1. Expand Knowledge Base

- Add more herb entries (target: 1,000+ herbs)
- Include herb-herb interaction knowledge
- Add contraindication information

### 2. Fine-tune Retrieval

- Train custom embedding model on TCM data
- Use herb category as additional filter
- Implement re-ranking based on herb frequency

### 3. Multi-modal RAG

- Add symptom-disease knowledge
- Add disease-herb relationships
- Combine multiple knowledge sources

### 4. Evaluation

- Run larger test set (100+ samples)
- Compare with baseline on same samples
- Analyze failure cases

### 5. Production Optimization

- Cache embeddings for faster inference
- Reduce prompt size with knowledge summarization
- Implement fallback for retrieval failures

## References

1. **Dataset**: [PTM - Prescription as Topic Model](https://github.com/yao8839836/PTM)
2. **Embeddings**: [Sentence Transformers](https://www.sbert.net/)
3. **Model**: [DeepSeek API](https://platform.deepseek.com)
4. **RAG**: [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)

## Citation

```bibtex
@misc{ptm_herb_recommender_rag,
  title={TCM Herb Recommendation with RAG Enhancement},
  author={Your Name},
  year={2025},
  note={RAG-enhanced model with vocabulary constraint}
}
```

## Disclaimer

⚠️ **Medical Disclaimer**: This system is for **research and educational purposes only**. TCM herb recommendations should NOT be used for actual medical treatment. Always consult qualified TCM practitioners.

⚠️ **RAG Limitations**: Retrieved knowledge may not be complete or accurate. Always verify herb properties with authoritative sources.

## Support

**Common Issues**:

1. **OpenAI API key not set**
   ```bash
   export OPENAI_API_KEY='your-openai-api-key'
   ```

2. **API rate limit exceeded**
   ```
   Error: Rate limit exceeded
   Solution: Wait a minute or upgrade to paid tier
   Current: Batches 100 herbs per request (4 batches total)
   ```

3. **High API cost**
   ```python
   # Cache embeddings after first run (see Performance Tips)
   # Embeddings cost: ~$0.02 for initial 337 herbs
   # Per-query cost: ~$0.000004 (very cheap)
   ```

4. **Low RAG improvement**
   ```python
   # Increase top-K to 15-20
   # Check retrieved knowledge relevance
   # Verify symptom descriptions are detailed enough
   ```

5. **Import Error**
   ```bash
   pip install openai numpy
   # Note: sentence-transformers NOT required!
   ```
