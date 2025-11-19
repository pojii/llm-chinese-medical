"""
Hybrid Knowledge Graph combining Chinese Medical KG and DRKG.
Provides enhanced retrieval from multiple knowledge sources.
"""
import json
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from knowledge_graph import MedicalKnowledgeGraph


class HybridMedicalKG:
    """
    Hybrid knowledge graph combining:
    1. Chinese Medical KG (from QASystemOnMedicalKG)
    2. DRKG-style drug-disease relationships

    This provides more comprehensive medical knowledge for predictions.
    """

    def __init__(
        self,
        chinese_kg_path: str = "./data/medical.json",
        drkg_path: str = None
    ):
        """
        Initialize hybrid knowledge graph.

        Args:
            chinese_kg_path: Path to Chinese medical KG
            drkg_path: Path to DRKG data (TSV format), None for sample data
        """
        # Load Chinese Medical KG
        print("Loading Chinese Medical Knowledge Graph...")
        self.chinese_kg = MedicalKnowledgeGraph(chinese_kg_path)

        # Initialize DRKG components
        self.drkg_compounds = {}  # compound_name -> properties
        self.drkg_diseases = {}   # disease_name -> properties
        self.drkg_relations = defaultdict(list)  # (entity1, relation_type) -> [entity2]

        # Load DRKG
        if drkg_path:
            self.load_drkg(drkg_path)
        else:
            self.load_sample_drkg()

        print(f"Hybrid KG initialized:")
        print(f"  - Chinese KG: {len(self.chinese_kg.entities)} entities")
        print(f"  - DRKG compounds: {len(self.drkg_compounds)}")
        print(f"  - DRKG diseases: {len(self.drkg_diseases)}")
        print(f"  - DRKG relations: {len(self.drkg_relations)}")

    def load_drkg(self, drkg_path: str):
        """
        Load DRKG from TSV file.

        Format: head_entity\trelation\ttail_entity
        """
        print(f"Loading DRKG from {drkg_path}...")

        with open(drkg_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 3:
                    continue

                head, relation, tail = parts

                # Extract entity type and name
                # Format: EntityType::EntityName
                if '::' in head:
                    head_type, head_name = head.split('::', 1)
                else:
                    head_type, head_name = 'Unknown', head

                if '::' in tail:
                    tail_type, tail_name = tail.split('::', 1)
                else:
                    tail_type, tail_name = 'Unknown', tail

                # Store compounds and diseases
                if head_type == 'Compound':
                    self.drkg_compounds[head_name] = {'name': head_name, 'type': 'Compound'}
                elif head_type == 'Disease':
                    self.drkg_diseases[head_name] = {'name': head_name, 'type': 'Disease'}

                if tail_type == 'Compound':
                    self.drkg_compounds[tail_name] = {'name': tail_name, 'type': 'Compound'}
                elif tail_type == 'Disease':
                    self.drkg_diseases[tail_name] = {'name': tail_name, 'type': 'Disease'}

                # Store relation
                self.drkg_relations[(head, relation)].append(tail)

        print(f"DRKG loaded: {len(self.drkg_compounds)} compounds, {len(self.drkg_diseases)} diseases")

    def load_sample_drkg(self):
        """
        Load sample DRKG-style data for demonstration.
        Based on common Chinese medicines and their relationships.
        """
        print("Loading sample DRKG data...")

        # Sample drug-disease relationships (based on TCM knowledge)
        # Format: (compound, relation, disease/symptom)
        sample_relations = [
            # 感冒相关
            ("Compound::银翘解毒片", "TREATS", "Disease::感冒"),
            ("Compound::银翘解毒片", "TREATS", "Disease::发热"),
            ("Compound::银翘解毒片", "TREATS", "Disease::头痛"),
            ("Compound::连花清瘟胶囊", "TREATS", "Disease::感冒"),
            ("Compound::连花清瘟胶囊", "TREATS", "Disease::发热"),
            ("Compound::板蓝根颗粒", "TREATS", "Disease::感冒"),
            ("Compound::板蓝根颗粒", "TREATS", "Disease::咽喉肿痛"),

            # 消化系统
            ("Compound::藿香正气丸", "TREATS", "Disease::腹泻"),
            ("Compound::藿香正气丸", "TREATS", "Disease::腹痛"),
            ("Compound::藿香正气丸", "TREATS", "Disease::呕吐"),
            ("Compound::保和丸", "TREATS", "Disease::消化不良"),
            ("Compound::保和丸", "TREATS", "Disease::腹胀"),

            # 神经系统
            ("Compound::安神补脑液", "TREATS", "Disease::失眠"),
            ("Compound::安神补脑液", "TREATS", "Disease::健忘"),
            ("Compound::天王补心丹", "TREATS", "Disease::失眠"),
            ("Compound::天王补心丹", "TREATS", "Disease::心悸"),
            ("Compound::天王补心丹", "TREATS", "Disease::多梦"),

            # 肝系统
            ("Compound::龙胆泻肝丸", "TREATS", "Disease::肝火旺盛"),
            ("Compound::龙胆泻肝丸", "TREATS", "Disease::口苦"),
            ("Compound::龙胆泻肝丸", "TREATS", "Disease::目赤肿痛"),
            ("Compound::逍遥丸", "TREATS", "Disease::肝郁"),
            ("Compound::逍遥丸", "TREATS", "Disease::月经不调"),

            # 肾系统
            ("Compound::六味地黄丸", "TREATS", "Disease::肾阴虚"),
            ("Compound::六味地黄丸", "TREATS", "Disease::腰膝酸软"),
            ("Compound::金匮肾气丸", "TREATS", "Disease::肾阳虚"),

            # 副作用关系
            ("Compound::银翘解毒片", "HAS_SIDE_EFFECT", "Disease::轻度胃肠道反应"),
            ("Compound::龙胆泻肝丸", "HAS_SIDE_EFFECT", "Disease::脾胃虚弱"),

            # 相互作用
            ("Compound::银翘解毒片", "INTERACTS_WITH", "Compound::阿司匹林"),
            ("Compound::六味地黄丸", "ENHANCES", "Compound::金匮肾气丸"),
        ]

        for head, relation, tail in sample_relations:
            # Extract entity names
            head_type, head_name = head.split('::')
            tail_type, tail_name = tail.split('::')

            # Store entities
            if head_type == 'Compound':
                self.drkg_compounds[head_name] = {
                    'name': head_name,
                    'type': 'Compound',
                    'source': 'TCM'
                }
            elif head_type == 'Disease':
                self.drkg_diseases[head_name] = {
                    'name': head_name,
                    'type': 'Disease',
                    'source': 'TCM'
                }

            if tail_type == 'Compound':
                self.drkg_compounds[tail_name] = {
                    'name': tail_name,
                    'type': 'Compound',
                    'source': 'TCM'
                }
            elif tail_type == 'Disease':
                self.drkg_diseases[tail_name] = {
                    'name': tail_name,
                    'type': 'Disease',
                    'source': 'TCM'
                }

            # Store relation
            self.drkg_relations[(head, relation)].append(tail)

        print(f"Sample DRKG loaded: {len(self.drkg_compounds)} compounds, {len(self.drkg_diseases)} diseases")

    def find_treatments(self, symptoms: List[str]) -> Dict[str, float]:
        """
        Find treatments from both KGs for given symptoms.

        Args:
            symptoms: List of symptom descriptions

        Returns:
            Dictionary mapping medicine names to confidence scores
        """
        treatments = defaultdict(float)

        # Search in Chinese KG
        for symptom in symptoms:
            # Direct symptom lookup
            diseases = self.chinese_kg.find_diseases_by_symptom(symptom)
            for disease in diseases:
                disease_info = self.chinese_kg.get_disease_info(disease)
                if disease_info and 'drug' in disease_info:
                    drugs = disease_info['drug']
                    if isinstance(drugs, list):
                        for drug in drugs:
                            treatments[drug] += 0.3  # Weight for Chinese KG

        # Search in DRKG
        for symptom in symptoms:
            # Look for disease-compound relationships
            disease_key = f"Disease::{symptom}"

            for (entity, relation), targets in self.drkg_relations.items():
                # Find compounds that treat this symptom
                if relation == "TREATS":
                    for target in targets:
                        if symptom in target or disease_key == target:
                            # Extract compound name
                            if entity.startswith("Compound::"):
                                compound_name = entity.split("::", 1)[1]
                                treatments[compound_name] += 0.7  # Higher weight for DRKG

        return dict(sorted(treatments.items(), key=lambda x: x[1], reverse=True))

    def get_drug_info(self, drug_name: str) -> Dict:
        """
        Get comprehensive drug information from both KGs.

        Args:
            drug_name: Name of the drug

        Returns:
            Dictionary with drug information
        """
        info = {
            'name': drug_name,
            'chinese_kg': None,
            'drkg': None,
            'treats': [],
            'side_effects': [],
            'interactions': []
        }

        # Check Chinese KG
        # Note: Chinese KG doesn't have direct drug lookup, but we can search

        # Check DRKG
        if drug_name in self.drkg_compounds:
            info['drkg'] = self.drkg_compounds[drug_name]

            # Find what it treats
            compound_key = f"Compound::{drug_name}"
            for (entity, relation), targets in self.drkg_relations.items():
                if entity == compound_key:
                    if relation == "TREATS":
                        info['treats'].extend([t.split("::")[-1] for t in targets])
                    elif relation == "HAS_SIDE_EFFECT":
                        info['side_effects'].extend([t.split("::")[-1] for t in targets])
                    elif relation in ["INTERACTS_WITH", "ENHANCES"]:
                        info['interactions'].extend([t.split("::")[-1] for t in targets])

        return info

    def search_hybrid_context(self, query: str, top_k: int = 3) -> str:
        """
        Search for relevant context from both KGs.

        Args:
            query: Search query
            top_k: Number of top results

        Returns:
            Formatted context string
        """
        context_parts = []

        # Get context from Chinese KG
        chinese_context = self.chinese_kg.search_relevant_context(query, top_k=top_k)
        if chinese_context and chinese_context != "未找到相关医学知识。":
            context_parts.append(f"【中医知识库】\n{chinese_context}")

        # Extract symptoms/diseases from query
        symptoms = []
        for word in ['头痛', '发热', '咳嗽', '流鼻涕', '腹痛', '腹泻', '失眠', '心悸', '口苦', '口干']:
            if word in query:
                symptoms.append(word)

        # Find treatments from DRKG
        if symptoms:
            treatments = self.find_treatments(symptoms)
            if treatments:
                drkg_context = "【药物知识图谱推荐】\n"
                for drug, score in list(treatments.items())[:3]:
                    drug_info = self.get_drug_info(drug)
                    drkg_context += f"\n药物: {drug} (置信度: {score:.2f})\n"
                    if drug_info['treats']:
                        drkg_context += f"  适应症: {', '.join(drug_info['treats'][:3])}\n"
                    if drug_info['side_effects']:
                        drkg_context += f"  注意: {', '.join(drug_info['side_effects'][:2])}\n"

                context_parts.append(drkg_context)

        return "\n\n".join(context_parts) if context_parts else "未找到相关医学知识。"


if __name__ == "__main__":
    # Test hybrid KG
    print("=" * 80)
    print("Testing Hybrid Medical Knowledge Graph")
    print("=" * 80)

    hybrid_kg = HybridMedicalKG(chinese_kg_path="../data/medical.json")

    # Test 1: Find treatments for symptoms
    print("\n测试1: 查找治疗方案")
    print("-" * 80)
    symptoms = ['头痛', '发热']
    treatments = hybrid_kg.find_treatments(symptoms)
    print(f"症状: {symptoms}")
    print(f"推荐药物:")
    for drug, score in treatments.items():
        print(f"  {drug}: {score:.2f}")

    # Test 2: Get drug info
    print("\n测试2: 获取药物信息")
    print("-" * 80)
    drug_info = hybrid_kg.get_drug_info("银翘解毒片")
    print(f"药物: {drug_info['name']}")
    print(f"治疗: {drug_info['treats']}")
    print(f"副作用: {drug_info['side_effects']}")
    print(f"相互作用: {drug_info['interactions']}")

    # Test 3: Hybrid context search
    print("\n测试3: 混合知识检索")
    print("-" * 80)
    query = "患者症状: 头痛, 发热。请推荐合适的中药。"
    context = hybrid_kg.search_hybrid_context(query, top_k=2)
    print(f"查询: {query}")
    print(f"\n检索结果:\n{context}")
