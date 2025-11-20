"""
Recommender Predictor Wrapper for Medical Recommendation.
Wraps DeepSeek API to return ranked lists of drug recommendations.
"""
import os
from typing import List, Optional
from openai import OpenAI
from recommender_dataset import extract_drugs_from_text


class MedicalRecommenderPredictor:
    """
    Medical Recommender using DeepSeek API.

    Returns ranked lists of drug/treatment recommendations.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        max_recommendations: int = 50
    ):
        """
        Initialize recommender predictor.

        Args:
            api_key: DeepSeek API key (default: from DEEPSEEK_API_KEY env var)
            model: Model name (default: deepseek-chat)
            max_recommendations: Maximum number of recommendations to return
        """
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.model = model
        self.max_recommendations = max_recommendations

        if not self.api_key:
            raise ValueError("API key required. Set DEEPSEEK_API_KEY environment variable.")

        # Initialize OpenAI client with DeepSeek
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )

        print(f"Initialized Medical Recommender with model: {self.model}")

    def recommend_without_kg(self, query: str, top_k: int = 50) -> List[str]:
        """
        Recommend drugs without knowledge graph context.

        Args:
            query: Medical query (symptoms or disease)
            top_k: Number of recommendations to return

        Returns:
            Ranked list of drug names
        """
        prompt = f"""你是一个专业的医学助手。请根据以下查询推荐适合的药物或治疗方法。

查询：{query}

要求：
1. 请直接列出推荐的药物名称，每个药物用顿号（、）分隔
2. 按照推荐优先级从高到低排序
3. 只输出药物名称，不要输出任何解释或说明
4. 尽可能多推荐一些药物（至少10个）

推荐药物："""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的中医药助手，擅长根据症状和疾病推荐药物。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                top_p=0.85,
                max_tokens=500
            )

            result_text = response.choices[0].message.content.strip()

            # Extract drugs from response
            drugs = extract_drugs_from_text(result_text)

            # Return top K
            return drugs[:top_k]

        except Exception as e:
            print(f"Error in recommendation: {e}")
            return []

    def recommend_with_kg(self, query: str, kg_context: str, top_k: int = 50) -> List[str]:
        """
        Recommend drugs with knowledge graph context.

        Args:
            query: Medical query (symptoms or disease)
            kg_context: Knowledge graph context
            top_k: Number of recommendations to return

        Returns:
            Ranked list of drug names
        """
        prompt = f"""你是一个专业的医学助手。请根据以下查询和知识图谱信息推荐适合的药物或治疗方法。

【知识图谱信息】
{kg_context}

【患者查询】
{query}

要求：
1. 优先参考知识图谱中的药物信息
2. 请直接列出推荐的药物名称，每个药物用顿号（、）分隔
3. 按照推荐优先级从高到低排序
4. 只输出药物名称，不要输出任何解释或说明
5. 尽可能多推荐一些药物（至少10个）

推荐药物："""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的中医药助手，擅长根据症状、疾病和医学知识推荐药物。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                top_p=0.85,
                max_tokens=500
            )

            result_text = response.choices[0].message.content.strip()

            # Extract drugs from response
            drugs = extract_drugs_from_text(result_text)

            # Return top K
            return drugs[:top_k]

        except Exception as e:
            print(f"Error in recommendation: {e}")
            return []


if __name__ == "__main__":
    # Test the recommender
    print("Testing Medical Recommender Predictor")
    print("=" * 80)

    # Initialize predictor
    try:
        predictor = MedicalRecommenderPredictor()

        # Test query
        query = "患者出现发热、咳嗽、咽痛的症状，请推荐适合的药物或治疗方法。"

        print(f"\nQuery: {query}")
        print("\n" + "─" * 80)
        print("Testing recommendation without KG...")
        print("─" * 80)

        recommendations = predictor.recommend_without_kg(query, top_k=10)

        print(f"\nTop 10 Recommendations:")
        for i, drug in enumerate(recommendations, 1):
            print(f"  {i}. {drug}")

        # Test with KG context
        kg_context = """疾病名称: 感冒
症状: 发热、咳嗽、咽痛、流鼻涕
推荐药物: 连花清瘟胶囊、板蓝根颗粒、银翘解毒片、感冒清热颗粒"""

        print("\n" + "─" * 80)
        print("Testing recommendation with KG...")
        print("─" * 80)
        print(f"\nKG Context: {kg_context[:100]}...")

        recommendations_kg = predictor.recommend_with_kg(query, kg_context, top_k=10)

        print(f"\nTop 10 Recommendations (with KG):")
        for i, drug in enumerate(recommendations_kg, 1):
            print(f"  {i}. {drug}")

    except ValueError as e:
        print(f"\nError: {e}")
        print("Please set DEEPSEEK_API_KEY environment variable to test.")
