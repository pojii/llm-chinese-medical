# Medical Recommender Dataset Structure

## Overview

`MedicalRecommenderDataset` ใช้สำหรับ medical recommendation task โดยแปลงข้อมูลจาก `medical.json` (Knowledge Graph) ให้เป็น recommendation samples

## Data Source

- **Source File**: `data/medical.json` (45MB)
- **Original Format**: Medical Knowledge Graph จาก QASystemOnMedicalKG
- **Total Entities**: ~8,808 diseases/conditions
- **Total Samples Generated**: **15,250 samples**
  - Disease-based queries: 7,645 samples
  - Symptom-based queries: 7,605 samples

## Sample Structure

แต่ละ sample มี **6 fields**:

```python
{
    'query_type': str,      # 'disease' หรือ 'symptom'
    'query': str,           # Query text สำหรับ LLM
    'disease_name': str,    # ชื่อโรค (ground truth disease)
    'ground_truth': set,    # Set of drug names (ยาที่ถูกต้อง)
    'entity': dict          # Entity เต็มจาก medical.json
}
```

### Field Descriptions

#### 1. `query_type` (str)
ประเภทของ query:
- `"disease"` - ใช้ชื่อโรคเป็น query
- `"symptom"` - ใช้อาการเป็น query

#### 2. `query` (str)
ข้อความที่จะใช้ query ต่อ LLM:
- **Disease-based**: ชื่อโรคโดยตรง เช่น `"百日咳"`, `"感冒"`
- **Symptom-based**: รายการอาการ เช่น `"发热、咳嗽、咽痛、流鼻涕、头痛"`

**Note**: ใน symptom-based จะเอาอาการ 5 ตัวแรก (ถ้ามีมากกว่า)

#### 3. `disease_name` (str)
ชื่อโรคที่สัมพันธ์กับ sample นี้ เช่น:
- `"百日咳"` (Whooping cough)
- `"感冒"` (Common cold)
- `"苯中毒"` (Benzene poisoning)

#### 4. `ground_truth` (set)
**ยาที่ถูกต้อง** สำหรับโรคนี้ (ground truth for evaluation)

**Source fields** จาก medical.json:
- `recommand_drug` - ยาที่แนะนำ
- `common_drug` - ยาที่ใช้บ่อย

**Format**: Python `set` (เพื่อไม่ให้ซ้ำ)

**Example**:
```python
{
    '红霉素肠溶片',
    '穿心莲内酯片',
    '琥乙红霉素颗粒',
    '百咳静糖浆',
    '琥乙红霉素片',
    '环酯红霉素片'
}
```

#### 5. `entity` (dict)
Entity เต็มจาก medical.json รวมข้อมูล:
- `name` - ชื่อโรค
- `desc` - คำอธิบาย
- `symptom` - อาการ
- `cause` - สาเหตุ
- `prevent` - การป้องกัน
- `recommand_drug` - ยาที่แนะนำ
- `common_drug` - ยาที่ใช้บ่อย
- `cure_way` - วิธีรักษา
- และอื่นๆ...

## Example Samples

### Example 1: Disease-based Query

```python
{
    'query_type': 'disease',
    'query': '百日咳',
    'disease_name': '百日咳',
    'ground_truth': {
        '红霉素肠溶片',
        '穿心莲内酯片',
        '琥乙红霉素颗粒',
        '百咳静糖浆',
        '琥乙红霉素片',
        '环酯红霉素片'
    },
    'entity': {
        'name': '百日咳',
        'desc': '百日咳是由百日咳杆菌所致的急性呼吸道传染病...',
        'symptom': ['吸气时有蝉鸣音', '痉挛性咳嗽', '胸闷', ...],
        'recommand_drug': ['琥乙红霉素片', '百咳静糖浆', ...],
        'common_drug': ['穿心莲内酯片', '红霉素肠溶片'],
        ...
    }
}
```

**Formatted Query** (ใช้กับ LLM):
```
患者诊断为百日咳，请推荐适合的药物或治疗方法。
```

### Example 2: Symptom-based Query

```python
{
    'query_type': 'symptom',
    'query': '吸气时有蝉鸣音、痉挛性咳嗽、胸闷、肺阴虚、抽搐',
    'disease_name': '百日咳',
    'ground_truth': {
        '红霉素肠溶片',
        '穿心莲内酯片',
        '琥乙红霉素颗粒',
        '百咳静糖浆',
        '琥乙红霉素片',
        '环酯红霉素片'
    },
    'entity': { ... }  # Same entity as above
}
```

**Formatted Query** (ใช้กับ LLM):
```
患者出现以下症状：吸气时有蝉鸣音、痉挛性咳嗽、胸闷、肺阴虚、抽搐，请推荐适合的药物或治疗方法。
```

## Dataset Statistics

### Size Distribution

```
Total samples: 15,250
├── Disease-based: 7,645 (50.1%)
└── Symptom-based: 7,605 (49.9%)
```

### Ground Truth Distribution

```
Drugs per sample:
  Minimum:  1 drug
  Maximum:  24 drugs
  Mean:     7.81 drugs
  Median:   6 drugs
```

**Distribution breakdown:**
- ~20% samples มี ≥10 drugs
- ~60% samples มี 5-9 drugs
- ~20% samples มี 1-4 drugs

## Data Loading

### Basic Usage

```python
from recommender_dataset import MedicalRecommenderDataset

# Load dataset
dataset = MedicalRecommenderDataset("../data/medical.json")
print(f"Total: {len(dataset)} samples")

# Get a sample
sample = dataset.get_sample(0)
print(f"Query Type: {sample['query_type']}")
print(f"Query: {sample['query']}")
print(f"Ground Truth: {sample['ground_truth']}")

# Format for LLM
query = dataset.format_query(sample)
print(f"LLM Query: {query}")
```

### Filtering Samples

```python
# Filter by query type
disease_indices = dataset.filter_by_query_type('disease')
symptom_indices = dataset.filter_by_query_type('symptom')

print(f"Disease samples: {len(disease_indices)}")
print(f"Symptom samples: {len(symptom_indices)}")

# Filter by minimum drugs
min_3_drugs = dataset.filter_by_min_drugs(min_drugs=3)
min_5_drugs = dataset.filter_by_min_drugs(min_drugs=5)
min_10_drugs = dataset.filter_by_min_drugs(min_drugs=10)

print(f"Samples with ≥3 drugs: {len(min_3_drugs)}")
print(f"Samples with ≥5 drugs: {len(min_5_drugs)}")
print(f"Samples with ≥10 drugs: {len(min_10_drugs)}")
```

### Accessing Fields

```python
sample = dataset.get_sample(0)

# Basic fields
query_type = sample['query_type']        # 'disease' or 'symptom'
query = sample['query']                  # Query text
disease = sample['disease_name']         # Disease name
drugs = sample['ground_truth']           # Set of drug names

# Full entity data
entity = sample['entity']
description = entity.get('desc', '')
symptoms = entity.get('symptom', [])
cause = entity.get('cause', '')
prevention = entity.get('prevent', '')
```

## Query Formatting

Dataset มี helper function สำหรับ format query:

```python
def format_query(sample):
    if sample['query_type'] == 'disease':
        return f"患者诊断为{sample['query']}，请推荐适合的药物或治疗方法。"
    else:  # symptom
        return f"患者出现以下症状：{sample['query']}，请推荐适合的药物或治疗方法。"
```

## Drug Extraction

สำหรับ extract drug names จาก LLM response:

```python
from recommender_dataset import extract_drugs_from_text

response = "建议使用银翘解毒片、连花清瘟胶囊和板蓝根颗粒进行治疗。"
drugs = extract_drugs_from_text(response)
# Output: ['银翘解毒片', '连花清瘟胶囊', '板蓝根颗粒']
```

**Features:**
- Remove parentheses and brackets
- Split by Chinese delimiters (、，；等)
- Remove common prefixes: "建议使用", "推荐使用", etc.
- Remove common suffixes: "进行治疗", "治疗", "等"
- Remove numbering: "1.", "2、"
- Filter out common phrases
- Preserve order (for ranking)
- Remove duplicates

## Data Quality

### Source Reliability
- ✅ Data จาก **real medical knowledge graph** (QASystemOnMedicalKG)
- ✅ Drug names จากฐานข้อมูลจริง (recommand_drug, common_drug)
- ✅ Not AI-generated (ต่างจาก NER dataset ตัวเก่า)

### Coverage
- **Diseases**: 7,645+ diseases/conditions
- **Drugs**: หลายพันรายการ (unique drug names)
- **Symptoms**: หลายร้อยประเภท

### Limitations
- บางโรคมียาน้อย (minimum 1 drug)
- บางโรคมียามาก (maximum 24 drugs) - อาจจะ noisy
- Symptom-based queries ใช้แค่ 5 อาการแรก
- ไม่มีข้อมูล contraindications หรือ drug interactions

## Use Cases

### 1. Recommender System Evaluation
ใช้เป็น test set สำหรับวัด Precision@K, Recall@K, MAP@K, MRR@K

### 2. Knowledge Graph Enhancement Testing
เปรียบเทียบ LLM with/without KG augmentation

### 3. Ranking Model Training
ใช้เป็น training data สำหรับ learning-to-rank models

### 4. Medical NLP Research
ศึกษาการ recommend ยาจาก symptoms/disease names

## Related Files

- `recommender_dataset.py` - Dataset loader
- `recommender_predictor.py` - Predictor wrapper
- `recommender_metrics.py` - Evaluation metrics
- `main_recommender.py` - Full evaluation pipeline
- `medical.json` - Source knowledge graph data

## Citation

```bibtex
@misc{qasystemonmedicalkg,
  title={QASystemOnMedicalKG},
  author={Chen, Zhihao},
  year={2021},
  publisher={GitHub},
  url={https://github.com/zhihao-chen/QASystemOnMedicalKG}
}
```

## Notes

- Dataset จะ **auto-load** เมื่อสร้าง `MedicalRecommenderDataset()`
- ใช้เวลา ~2-3 วินาทีในการ load และ parse
- Memory usage: ~100-200MB (เพราะเก็บ full entity dict)
- Thread-safe สำหรับ read operations
