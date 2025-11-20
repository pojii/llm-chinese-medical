# PTM Dataset - Traditional Chinese Medicine Herbs Analysis

## Dataset Overview

**Source**: [PTM (Prescription as Topic Model)](https://github.com/yao8839836/PTM)

**Dataset Statistics:**
```
Total prescriptions: 98,334
Total herb mentions: 606,162
Unique herbs: ~1,000-2,000 (estimated after cleaning)
Average herbs per prescription: 6.16
```

**Task**: Symptom → Herb recommendation (Traditional Chinese Medicine)

## Top 50 Most Common Herbs (按频率排序)

| Rank | Chinese Name | English Name | Count | Percentage | Category |
|------|-------------|-------------|--------|-----------|----------|
| 1 | 甘草 | Licorice Root | 18,380 | 3.03% | Tonic/Harmonizing |
| 2 | 当归 | Chinese Angelica | 11,630 | 1.92% | Blood tonic |
| 3 | 人参 | Ginseng | 11,268 | 1.86% | Qi tonic |
| 4 | 白术 | White Atractylodes | 8,029 | 1.32% | Qi tonic |
| 5 | 防风 | Siler Root | 6,438 | 1.06% | Wind-dispelling |
| 6 | 黄芩 | Scutellaria Root | 6,197 | 1.02% | Heat-clearing |
| 7 | 木香 | Costus Root | 5,628 | 0.93% | Qi-regulating |
| 8 | 川芎 | Chuanxiong | 5,616 | 0.93% | Blood-moving |
| 9 | 茯苓 | Poria | 5,288 | 0.87% | Dampness-draining |
| 10 | 黄连 | Coptis | 5,205 | 0.86% | Heat-clearing |
| 11 | 陈皮 | Tangerine Peel | 4,867 | 0.80% | Qi-regulating |
| 12 | 半夏 | Pinellia | 4,645 | 0.77% | Phlegm-transforming |
| 13 | 干姜 | Dried Ginger | 4,349 | 0.72% | Warming |
| 14 | 大黄 | Rhubarb | 4,240 | 0.70% | Purgative |
| 15 | 黄耆 | Astragalus | 4,222 | 0.70% | Qi tonic |
| 16 | 柴胡 | Bupleurum | 4,007 | 0.66% | Harmonizing |
| 17 | 枳壳 | Bitter Orange | 4,007 | 0.66% | Qi-regulating |
| 18 | 附子 | Aconite | 3,755 | 0.62% | Warming |
| 19 | 桔梗 | Platycodon | 3,752 | 0.62% | Phlegm-transforming |
| 20 | 桂心 | Cinnamon Heart | 3,605 | 0.59% | Warming |
| 21 | 槟榔 | Areca Seed | 3,455 | 0.57% | Qi-moving |
| 22 | 羌活 | Notopterygium | 3,388 | 0.56% | Wind-dispelling |
| 23 | 麝香 | Musk | 3,358 | 0.55% | Resuscitating |
| 24 | 白茯苓 | White Poria | 3,223 | 0.53% | Dampness-draining |
| 25 | 厚朴 | Magnolia Bark | 3,207 | 0.53% | Qi-regulating |
| 26 | 细辛 | Asarum | 3,171 | 0.52% | Warm/Acrid |
| 27 | 白芷 | Angelica Dahurica | 3,118 | 0.51% | Wind-dispelling |
| 28 | 白芍 | White Peony | 2,845 | 0.47% | Blood-nourishing |
| 29 | 杏仁 | Apricot Kernel | 2,841 | 0.47% | Cough-relieving |
| 30 | 乳香 | Frankincense | 2,682 | 0.44% | Blood-moving |
| 31 | 芍药 | Peony | 2,651 | 0.44% | Blood-nourishing |
| 32 | 黄柏 | Phellodendron | 2,600 | 0.43% | Heat-clearing |
| 33 | 麻黄 | Ephedra | 2,551 | 0.42% | Wind-cold releasing |
| 34 | 麦门冬 | Ophiopogon | 2,486 | 0.41% | Yin-nourishing |
| 35 | 苍术 | Atractylodes | 2,336 | 0.39% | Dampness-drying |
| 36 | 雄黄 | Realgar | 2,290 | 0.38% | Toxic/External |
| 37 | 石膏 | Gypsum | 2,288 | 0.38% | Heat-clearing |
| 38 | 生地 | Rehmannia Fresh | 2,252 | 0.37% | Heat-clearing/Blood-cooling |
| 39 | 升麻 | Cimicifuga | 2,231 | 0.37% | Upbearing |
| 40 | 牛膝 | Achyranthes | 2,223 | 0.37% | Blood-moving |
| 41 | 知母 | Anemarrhena | 2,201 | 0.36% | Heat-clearing |
| 42 | 没药 | Myrrh | 2,168 | 0.36% | Blood-moving |
| 43 | 朱砂 | Cinnabar | 2,123 | 0.35% | Spirit-calming |
| 44 | 赤茯苓 | Red Poria | 2,108 | 0.35% | Dampness-draining |
| 45 | 木通 | Akebia | 2,084 | 0.34% | Damp-heat clearing |
| 46 | 五味子 | Schisandra | 2,083 | 0.34% | Astringent |
| 47 | 赤芍药 | Red Peony | 2,056 | 0.34% | Blood-cooling |
| 48 | 芎? | Chuanxiong (variant) | 2,015 | 0.33% | Blood-moving |
| 49 | 连翘 | Forsythia | 2,003 | 0.33% | Heat-clearing |
| 50 | 丁香 | Clove | 1,953 | 0.32% | Warming |

## Key Insights

### 1. Most Popular Herb: **甘草 (Licorice Root)**
- **18,380 mentions** (3.03% of all herbs)
- Used as a harmonizing agent in TCM
- Often combined with other herbs to reduce toxicity
- Appears in ~18.7% of prescriptions

### 2. Top Herb Categories

**Most Common Functions:**
1. **Qi Tonics** (补气药): 人参, 白术, 黄耆 - 23,519 mentions
2. **Heat-Clearing** (清热药): 黄芩, 黄连, 黄柏, 石膏 - 16,290 mentions
3. **Blood Tonics** (补血药): 当归, 白芍, 芍药, 生地 - 19,378 mentions
4. **Wind-Dispelling** (祛风药): 防风, 羌活, 白芷 - 12,944 mentions

### 3. Herb Frequency Distribution

```
Frequency Range          Number of Herbs
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
10,000+                  3 herbs (甘草, 当归, 人参)
5,000-9,999              7 herbs
1,000-4,999              86 herbs
500-999                  92 herbs
100-499                  373 herbs
10-99                    2,378 herbs
```

**Coverage Analysis:**
- **Top 20 herbs** account for **20.64%** of all herb mentions
- **Top 50 herbs** account for **33.18%** of all herb mentions
- **Top 100 herbs** likely cover **45-50%** of all uses

### 4. Herb Characteristics

**Warming Herbs (温热药):**
- 干姜 (Dried Ginger) - 4,349
- 附子 (Aconite) - 3,755
- 桂心 (Cinnamon) - 3,605
- 细辛 (Asarum) - 3,171

**Cooling Herbs (寒凉药):**
- 黄芩 (Scutellaria) - 6,197
- 黄连 (Coptis) - 5,205
- 大黄 (Rhubarb) - 4,240
- 石膏 (Gypsum) - 2,288

**Tonifying Herbs (补益药):**
- 人参 (Ginseng) - 11,268
- 当归 (Angelica) - 11,630
- 白术 (Atractylodes) - 8,029
- 黄耆 (Astragalus) - 4,222

## Data Quality Issues

### 1. Parsing Challenges
- Some entries contain multiple herbs in comma-separated format
- Mixed use of space, comma (、), and Chinese comma (，) as separators
- Some entries include full prescriptions with dosages embedded

### 2. Herb Variations
Examples of the same herb with different names:
- 茯苓 vs 白茯苓 vs 赤茯苓
- 芍药 vs 白芍 vs 赤芍药
- 川芎 vs 芎?

### 3. Data Noise
- **~90,000 unique entries** detected (too high)
- Many are actually combinations or full prescriptions
- After proper cleaning, estimate **~1,000-2,000** true unique herbs

## Prescription Format

**Original Format:**
```
症状描述 \t 药物配方
(Symptoms) \t (Herbs with dosages)
```

**Example:**
```
豆疮黑陷，或变紫暗色，证在急危者。\t 穿山甲（汤浸透，取甲锉碎，同热灰铛内慢火炒令黄色）五钱  红色曲（炒）  川乌（一枚，灰火中带焦炮）各二钱半
```

**Components:**
- Symptom: "豆疮黑陷，或变紫暗色，证在急危者" (Smallpox with dark spots)
- Herbs: 穿山甲 (Pangolin scale), 红色曲 (Red yeast rice), 川乌 (Aconite)
- Dosage: 五钱 (5 qian), 二钱半 (2.5 qian)
- Preparation: (汤浸透...) - steeping method

## Recommendation Task

### Input → Output Mapping

**Input**: Symptom description (症状)
- Example: "头痛发热咳嗽" (Headache, fever, cough)

**Output**: Ranked list of herbs (药材)
- Example: [麻黄, 桂枝, 杏仁, 甘草, 生姜]

### Task Difficulty

**Challenge Level: High**
- Large herb vocabulary (~1,000-2,000 herbs)
- Average 6.16 herbs per prescription
- Must learn symptom-herb associations
- Consider herb interactions and contraindications

**Compared to Modern Medicine:**
- Modern drug dataset: ~3,800 drugs (medical.json)
- TCM herb dataset: ~1,000-2,000 herbs (PTM)
- TCM has more emphasis on combinations

## Recommended K Values for Evaluation

Based on average 6.16 herbs per prescription:

```
K = 5   → Below average (strict precision)
K = 10  → ~1.6x average (balanced)
K = 15  → ~2.4x average
K = 20  → ~3.2x average (high recall)
K = 30  → ~4.9x average (very high recall)
```

**Suggested K values**: **5, 10, 15, 20, 30**

## File Locations

```
data/PTM/
├── data/
│   ├── prescriptions.txt (18MB)          # Main dataset: 98,334 prescriptions
│   ├── herbs_list.txt (11KB)             # Herb vocabulary
│   ├── pre_herbs.txt (907KB)             # Preprocessed herbs (indexed)
│   ├── pre_symptoms.txt (191KB)          # Preprocessed symptoms (indexed)
│   ├── herb_symptom_knowledge.txt        # Herb-symptom KB
│   └── herb_frequency_analysis.txt       # Our analysis results
```

## Next Steps

### 1. Dataset Preparation
- [ ] Improve herb name extraction
- [ ] Normalize herb variations (茯苓 = 白茯苓 = 赤茯苓)
- [ ] Create clean herb vocabulary (~1,000 herbs)
- [ ] Map to herb categories

### 2. Model Development
- [ ] Baseline: LLM without KG (症状 → 中药)
- [ ] Enhanced: LLM with herb knowledge
- [ ] Evaluation: P@K, R@K, MAP@K, MRR@K

### 3. Knowledge Enhancement
- [ ] Add herb properties (warm/cool, toxic/non-toxic)
- [ ] Add herb categories (补气/清热/活血)
- [ ] Add contraindications
- [ ] Add herb-herb interactions

## References

1. **Dataset**: [PTM - Prescription as Topic Model](https://github.com/yao8839836/PTM)
2. **Paper**: Topic Modeling for Traditional Chinese Medicine Prescriptions
3. **License**: Research use only (non-commercial)

## Usage Restrictions

⚠️ **Important**: This dataset is for **research purposes only**. Commercial use is prohibited.

⚠️ **Medical Disclaimer**: TCM herb recommendations are for educational research only. Always consult qualified TCM practitioners for actual treatment.
