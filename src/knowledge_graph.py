"""
Knowledge Graph module for loading and querying medical knowledge.
"""
import json
from typing import Dict, List, Optional
from collections import defaultdict


class MedicalKnowledgeGraph:
    """Medical Knowledge Graph for storing and querying medical entities."""

    def __init__(self, data_path: str = "./data/medical.json"):
        """
        Initialize the knowledge graph from medical.json

        Args:
            data_path: Path to the medical knowledge JSON file
        """
        self.data_path = data_path
        self.entities = []
        self.disease_map = {}
        self.symptom_map = defaultdict(list)
        self.drug_map = defaultdict(list)
        self.load_data()

    def load_data(self):
        """Load medical knowledge from JSON file."""
        print(f"Loading knowledge graph from {self.data_path}...")

        with open(self.data_path, 'r', encoding='utf-8') as f:
            # The file contains multiple JSON objects, one per line
            content = f.read()
            # Try to parse as JSON array or JSONL format
            try:
                # Try as single JSON object
                data = json.loads(content)
                if not isinstance(data, list):
                    data = [data]
            except json.JSONDecodeError:
                # Try as JSONL (one JSON per line)
                data = []
                for line in content.strip().split('\n'):
                    if line.strip():
                        try:
                            data.append(json.loads(line))
                        except:
                            continue

        self.entities = data

        # Build indices for quick lookup
        for entity in self.entities:
            name = entity.get('name', '')

            # Map disease names
            self.disease_map[name] = entity

            # Extract symptoms if available
            if 'symptom' in entity:
                symptoms = entity['symptom']
                if isinstance(symptoms, str):
                    # Parse symptom string
                    for symptom in symptoms.split('，'):
                        symptom = symptom.strip()
                        if symptom:
                            self.symptom_map[symptom].append(name)

            # Extract drugs/treatments if available
            if 'drug' in entity or 'cure_way' in entity:
                drugs = entity.get('drug', [])
                if isinstance(drugs, list):
                    for drug in drugs:
                        if drug:
                            self.drug_map[drug].append(name)

        print(f"Loaded {len(self.entities)} entities")
        print(f"Indexed {len(self.disease_map)} diseases")
        print(f"Indexed {len(self.symptom_map)} symptoms")
        print(f"Indexed {len(self.drug_map)} drugs")

    def get_disease_info(self, disease_name: str) -> Optional[Dict]:
        """
        Get detailed information about a disease.

        Args:
            disease_name: Name of the disease

        Returns:
            Dictionary containing disease information
        """
        return self.disease_map.get(disease_name)

    def find_diseases_by_symptom(self, symptom: str) -> List[str]:
        """
        Find diseases associated with a symptom.

        Args:
            symptom: Symptom description

        Returns:
            List of disease names
        """
        return self.symptom_map.get(symptom, [])

    def find_diseases_by_drug(self, drug: str) -> List[str]:
        """
        Find diseases that can be treated with a drug.

        Args:
            drug: Drug name

        Returns:
            List of disease names
        """
        return self.drug_map.get(drug, [])

    def search_relevant_context(self, query: str, top_k: int = 3) -> str:
        """
        Search for relevant medical context based on query.

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            Formatted context string
        """
        # Simple keyword-based search
        relevant_entities = []

        for entity in self.entities:
            # Check if query keywords appear in entity fields
            entity_text = json.dumps(entity, ensure_ascii=False).lower()
            if any(keyword.lower() in entity_text for keyword in query.split()):
                relevant_entities.append(entity)
                if len(relevant_entities) >= top_k:
                    break

        # Format context
        context_parts = []
        for entity in relevant_entities:
            name = entity.get('name', 'Unknown')
            desc = entity.get('desc', '')
            cause = entity.get('cause', '')
            prevent = entity.get('prevent', '')
            symptom = entity.get('symptom', '')

            context = f"疾病名称: {name}\n"
            if desc:
                context += f"描述: {desc[:200]}...\n"
            if symptom:
                context += f"症状: {symptom[:100]}...\n"
            if cause:
                context += f"病因: {cause[:100]}...\n"
            if prevent:
                context += f"预防: {prevent[:100]}...\n"

            context_parts.append(context)

        return "\n---\n".join(context_parts) if context_parts else "未找到相关医学知识。"

    def get_treatment_recommendations(self, disease_name: str) -> str:
        """
        Get treatment recommendations for a disease.

        Args:
            disease_name: Name of the disease

        Returns:
            Treatment recommendations
        """
        disease_info = self.get_disease_info(disease_name)
        if not disease_info:
            return "未找到该疾病信息。"

        recommendations = []

        if 'cure_way' in disease_info:
            recommendations.append(f"治疗方法: {disease_info['cure_way'][:200]}")

        if 'drug' in disease_info:
            drugs = disease_info['drug']
            if isinstance(drugs, list):
                recommendations.append(f"推荐药物: {', '.join(drugs[:5])}")
            else:
                recommendations.append(f"推荐药物: {drugs}")

        if 'prevent' in disease_info:
            recommendations.append(f"预防措施: {disease_info['prevent'][:200]}")

        return "\n".join(recommendations) if recommendations else "暂无治疗建议。"


if __name__ == "__main__":
    # Test the knowledge graph
    kg = MedicalKnowledgeGraph()

    # Test disease lookup
    test_disease = list(kg.disease_map.keys())[0]
    print(f"\n测试疾病查询: {test_disease}")
    info = kg.get_disease_info(test_disease)
    print(f"疾病信息: {info.get('desc', '')[:100]}...")

    # Test context search
    print("\n测试上下文搜索: 感冒")
    context = kg.search_relevant_context("感冒")
    print(context[:300])
