"""
DeepSeek API predictor for medicine recommendations.
Uses DeepSeek API instead of loading models locally.
"""
import os
from typing import Optional

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = None


class DeepSeekAPIPredictor:
    """
    Predictor using DeepSeek API for medicine recommendations.
    More efficient than loading models locally.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek-chat"):
        """
        Initialize DeepSeek API predictor.

        Args:
            api_key: DeepSeek API key (if None, reads from DEEPSEEK_API_KEY env var)
            model: Model name (default: "deepseek-chat")
        """
        if not HAS_OPENAI:
            raise ImportError(
                "OpenAI library is required for DeepSeek API. "
                "Install it with: pip install openai"
            )

        # Get API key from environment if not provided
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DeepSeek API key not found. "
                "Please set DEEPSEEK_API_KEY environment variable or pass api_key parameter."
            )

        self.model = model

        # Initialize OpenAI client with DeepSeek base URL
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )

        print(f"Initialized DeepSeek API predictor with model: {model}")

    def predict_without_kg(self, query: str) -> str:
        """
        Predict medicine recommendation without KG using DeepSeek API.

        Args:
            query: Patient query with symptoms

        Returns:
            Medicine recommendation
        """
        messages = [
            {
                "role": "system",
                "content": "你是一位专业的中医药专家。根据患者症状，推荐1-2种最合适的中药。只返回药物名称，用顿号(、)分隔，不要添加任何其他解释或标点。"
            },
            {
                "role": "user",
                "content": query
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                top_p=0.85,
                max_tokens=30,
                stream=False
            )

            # Extract response
            result = response.choices[0].message.content.strip()

            # Clean up response - take only the first line
            result = result.split('\n')[0].strip()

            # Remove common prefixes
            for prefix in ["推荐中药：", "推荐：", "建议：", "可以服用"]:
                if result.startswith(prefix):
                    result = result[len(prefix):].strip()

            return result if result else "无法生成推荐"

        except Exception as e:
            print(f"Error during API call: {e}")
            return self._mock_prediction(query, use_kg=False)

    def predict_with_kg(self, query: str, kg_context: str) -> str:
        """
        Predict medicine recommendation with KG context using DeepSeek API.

        Args:
            query: Patient query with symptoms
            kg_context: Relevant medical knowledge from KG

        Returns:
            Medicine recommendation
        """
        # Truncate context if too long
        if len(kg_context) > 800:
            kg_context = kg_context[:800] + "..."

        messages = [
            {
                "role": "system",
                "content": "你是一位专业的中医药专家。根据提供的医学知识和患者症状，推荐1-2种最合适的中药。只返回药物名称，用顿号(、)分隔，不要添加任何其他解释或标点。"
            },
            {
                "role": "user",
                "content": f"""医学知识参考：
{kg_context}

患者情况：
{query}

请根据以上信息推荐中药："""
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                top_p=0.85,
                max_tokens=30,
                stream=False
            )

            # Extract response
            result = response.choices[0].message.content.strip()

            # Clean up response - take only the first line
            result = result.split('\n')[0].strip()

            # Remove common prefixes
            for prefix in ["推荐中药：", "推荐：", "建议：", "可以服用", "根据", "基于"]:
                if result.startswith(prefix):
                    result = result[len(prefix):].strip()

            return result if result else "无法生成推荐"

        except Exception as e:
            print(f"Error during API call: {e}")
            return self._mock_prediction(query, use_kg=True, context=kg_context)

    def _mock_prediction(self, query: str, use_kg: bool, context: str = None) -> str:
        """
        Mock prediction for fallback when API fails.

        Args:
            query: Patient query
            use_kg: Whether KG context was used
            context: KG context if available

        Returns:
            Mock medicine recommendation
        """
        # Simple mock based on keywords
        if "头痛" in query or "发热" in query:
            return "银翘解毒片"
        elif "咳嗽" in query:
            return "川贝枇杷膏"
        elif "失眠" in query:
            return "酸枣仁汤"
        elif "消化" in query or "腹泻" in query:
            return "保和丸"
        else:
            return "板蓝根颗粒"


if __name__ == "__main__":
    # Test the DeepSeek API predictor
    print("Testing DeepSeek API Predictor...")
    print("=" * 80)

    try:
        predictor = DeepSeekAPIPredictor()

        # Test queries
        queries = [
            "患者症状: 头痛, 发热。请推荐合适的中药。",
            "患者症状: 咳嗽, 流鼻涕，病因: 风寒感冒。请推荐合适的中药。",
            "患者症状: 失眠, 多梦, 心悸。请推荐合适的中药。"
        ]

        for i, query in enumerate(queries):
            print(f"\nTest {i+1}:")
            print(f"Query: {query}")
            result = predictor.predict_without_kg(query)
            print(f"Prediction: {result}")
            print("-" * 80)

    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\nTo use DeepSeek API, set your API key:")
        print("export DEEPSEEK_API_KEY='your-api-key-here'")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
