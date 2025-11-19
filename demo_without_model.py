"""
Demo script that runs without requiring torch/transformers.
Uses mock predictions to demonstrate the system.
"""
import sys
sys.path.insert(0, './src')

from knowledge_graph import MedicalKnowledgeGraph
from ner_dataset import TCMNERDataset


def main():
    """Run demo without LLM model."""
    print("""
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║                                                                            ║
    ║           中医药知识图谱演示 (无需LLM模型)                                     ║
    ║                  Knowledge Graph Demo (No Model)                          ║
    ║                                                                            ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    """)

    # Load Knowledge Graph
    print("\n[1/2] 加载医学知识图谱...")
    kg = MedicalKnowledgeGraph("./data/medical.json")

    # Load NER Dataset
    print("\n[2/2] 加载中医NER数据集...")
    dataset = TCMNERDataset()

    print("\n" + "=" * 80)
    print("知识图谱统计信息")
    print("=" * 80)
    print(f"实体总数: {len(kg.entities)}")
    print(f"疾病数量: {len(kg.disease_map)}")
    print(f"症状索引: {len(kg.symptom_map)}")
    print(f"药物索引: {len(kg.drug_map)}")

    # Demonstrate KG queries
    print("\n" + "=" * 80)
    print("知识图谱查询演示")
    print("=" * 80)

    # Example 1: Search by keyword
    print("\n【示例1】关键词搜索: '感冒'")
    print("-" * 80)
    context = kg.search_relevant_context("感冒", top_k=2)
    print(context)

    # Example 2: Get disease info
    if kg.entities:
        disease_name = kg.entities[0].get('name', '')
        print(f"\n【示例2】疾病详情查询: '{disease_name}'")
        print("-" * 80)
        disease_info = kg.get_disease_info(disease_name)
        if disease_info:
            print(f"名称: {disease_info.get('name', '')}")
            print(f"分类: {disease_info.get('category', '')}")
            desc = disease_info.get('desc', '')
            print(f"描述: {desc[:200]}...")

    # Demonstrate NER dataset
    print("\n" + "=" * 80)
    print("中医NER数据集演示")
    print("=" * 80)
    print(f"样本总数: {len(dataset)}")

    for i in range(min(3, len(dataset))):
        print(f"\n【样本 {i + 1}】")
        print("-" * 80)
        sample = dataset.get_sample(i)
        print(f"原文: {sample['text']}")

        entities = dataset.extract_entities(sample)
        print("\n提取的实体:")
        for entity_type, entity_list in entities.items():
            if entity_list:
                type_name = {
                    'SYM': '症状',
                    'CAU': '病因',
                    'HER': '药物',
                    'PRE': '处方',
                    'EFF': '功效'
                }.get(entity_type, entity_type)
                print(f"  {type_name}: {', '.join(entity_list)}")

        query = dataset.format_for_prediction(sample)
        print(f"\n生成的查询: {query}")

        # Use KG to provide context
        kg_context = kg.search_relevant_context(query, top_k=1)
        print(f"\n知识图谱检索结果:")
        print(kg_context[:300] + "..." if len(kg_context) > 300 else kg_context)

    # Simulate comparison
    print("\n" + "=" * 80)
    print("方法对比演示 (模拟)")
    print("=" * 80)

    sample = dataset.get_sample(0)
    query = dataset.format_for_prediction(sample)

    print(f"\n查询: {query}")

    entities = dataset.extract_entities(sample)
    ground_truth = entities.get('HER', [])
    print(f"\n真实标注: {ground_truth if ground_truth else '无'}")

    print(f"\n{'─' * 80}")
    print("【方法1】不使用知识图谱")
    print('─' * 80)
    print("说明: 仅依赖LLM的预训练知识")
    print("预测: 银翘解毒片、连花清瘟胶囊 (示例)")

    print(f"\n{'─' * 80}")
    print("【方法2】使用知识图谱增强")
    print('─' * 80)
    kg_context = kg.search_relevant_context(query, top_k=1)
    print(f"检索到的知识图谱上下文:")
    print(kg_context[:200] + "..." if len(kg_context) > 200 else kg_context)
    print("\n预测: 银翘解毒片 (示例)")
    print("说明: 结合知识图谱的专业医学知识，提供更准确的推荐")

    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print("""
✓ 知识图谱成功加载并可查询
✓ NER数据集可以提取症状、病因、药物等实体
✓ 系统可以将症状转换为查询并检索相关医学知识

要运行完整的LLM预测实验，请:
1. 安装依赖: pip install torch transformers
2. 运行: cd src && python main_comparison.py

或者使用mock模式进行快速演示 (不需要安装模型)。
    """)


if __name__ == "__main__":
    main()
