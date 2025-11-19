# Quick Start Guide

## 🚀 Quick Demo (No Installation Required)

Run the demo without installing PyTorch or Transformers:

```bash
python demo_without_model.py
```

This will demonstrate:
- ✅ Medical Knowledge Graph loading and querying
- ✅ TCM NER entity extraction
- ✅ Knowledge retrieval for symptoms
- ✅ System architecture overview

## 📋 System Test

Check if all components are working:

```bash
python test_system.py
```

## 🧪 Full Experiment (Requires Dependencies)

### Step 1: Install Dependencies

```bash
pip install torch transformers
```

**Note**: On CPU-only machines, use:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers
```

### Step 2: Run Comparison

```bash
cd src
python main_comparison.py
```

This will:
1. Load the medical knowledge graph (8000+ entities)
2. Initialize Chinese GPT-2 model (lightweight, CPU-compatible)
3. Run predictions on 5 sample cases
4. Compare results with and without knowledge graph
5. Save results to `outputs/comparison_results.json`

## 📊 Expected Output

```
Sample 1/5
================================================================================
原文: 患者出现头痛发热症状，建议服用银翘解毒片进行治疗。
查询: 患者症状: 头痛, 发热。请推荐合适的中药。
真实标注的药物: ['银翘解毒片']

【方法1】不使用知识图谱的LLM预测
────────────────────────────────────────────────────────────────
预测结果: 银翘解毒片、连花清瘟胶囊
推理时间: 0.15秒

【方法2】使用知识图谱增强的LLM预测
────────────────────────────────────────────────────────────────
检索到的知识图谱上下文:
疾病名称: 感冒
描述: 感冒是由病毒引起的上呼吸道感染...

预测结果: 银翘解毒片
推理时间: 0.18秒
```

## 🔧 Troubleshooting

### Issue: PyTorch installation fails

**Solution 1**: Use the demo without model
```bash
python demo_without_model.py
```

**Solution 2**: The system includes mock predictions as fallback. Even if torch is not installed, the main script will run with simulated results.

### Issue: Model download is slow

**Solution**: The first run downloads the Chinese GPT-2 model (~500MB). This is cached for future runs. Be patient on the first execution.

### Issue: Out of memory

**Solution**: The default model is already lightweight. If you still encounter issues, you can:
1. Reduce `num_samples` in `main_comparison.py` (line 196)
2. Use the demo mode instead

## 📖 Learn More

- See [README.md](README.md) for detailed documentation
- Check individual modules in `src/` directory
- Explore the medical knowledge graph in `data/medical.json`

## 🎯 Key Features Demonstrated

1. **Knowledge Graph Integration**
   - 8000+ medical entities indexed
   - Fast keyword-based retrieval
   - Disease, symptom, and drug mappings

2. **Named Entity Recognition**
   - BIO-format tagging
   - Extract symptoms, causes, herbs, effects
   - Generate structured queries

3. **LLM Prediction**
   - Lightweight Chinese model (CPU-compatible)
   - With/without KG comparison
   - Mock mode for quick testing

4. **Evaluation Framework**
   - Ground truth comparison
   - Timing measurements
   - Result persistence

## 💡 Next Steps

1. Try different queries in `demo_without_model.py`
2. Add your own TCM NER data
3. Experiment with different LLM models
4. Enhance the knowledge graph retrieval
5. Add evaluation metrics

## 📝 Project Structure

```
llm-chinese-medical/
├── demo_without_model.py      # 👈 START HERE (no deps required)
├── test_system.py             # System component test
├── src/
│   ├── knowledge_graph.py     # KG loader
│   ├── ner_dataset.py         # NER handler
│   ├── llm_predictor.py       # LLM prediction
│   └── main_comparison.py     # Full experiment
├── data/
│   └── medical.json           # Medical KG (45MB)
├── requirements.txt           # Dependencies
└── README.md                  # Full documentation
```

## ✨ Quick Tips

- 💻 Use `demo_without_model.py` for instant results
- 🔬 Use `main_comparison.py` for full experiment
- 📊 Check `outputs/` for saved results
- 🔍 Explore `data/medical.json` for KG structure
- 🎨 Customize NER samples in `ner_dataset.py`

Enjoy exploring the system! 🚀
