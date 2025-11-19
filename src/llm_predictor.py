"""
LLM-based medicine prediction system.
Uses lightweight models for CPU inference.
"""
from typing import Optional
import warnings

warnings.filterwarnings('ignore')

# Try to import torch and transformers, use mock if not available
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    pipeline = None


class MedicineLLMPredictor:
    """
    LLM-based predictor for medicine recommendations.
    Uses lightweight Chinese language models that can run on CPU.
    """

    def __init__(
        self,
        model_name: str = "uer/gpt2-chinese-cluecorpussmall",
        device: str = "cpu",
        max_length: int = 256
    ):
        """
        Initialize the LLM predictor.

        Args:
            model_name: Hugging Face model name (default: Chinese GPT2 small)
            device: Device to run on ('cpu' or 'cuda')
            max_length: Maximum generation length
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length

        print(f"Loading model: {model_name}...")
        print(f"Device: {device}")

        if not HAS_TORCH:
            print("Torch not available. Using mock predictor for demonstration...")
            self.model = None
            self.tokenizer = None
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32  # Use float32 for CPU
            ).to(device)

            self.model.eval()
            print("Model loaded successfully!")

        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to mock predictor for demonstration...")
            self.model = None
            self.tokenizer = None

    def predict_without_kg(self, query: str) -> str:
        """
        Predict medicine recommendation without knowledge graph context.

        Args:
            query: Patient query with symptoms

        Returns:
            Medicine recommendation
        """
        prompt = f"""根据患者症状推荐中药：

问题：{query}

推荐中药："""

        if self.model is None:
            # Mock response for demonstration
            return self._mock_prediction(query, use_kg=False)

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the generated part
            recommendation = response.split("推荐中药：")[-1].strip()

            return recommendation if recommendation else "无法生成推荐"

        except Exception as e:
            print(f"Error during prediction: {e}")
            return self._mock_prediction(query, use_kg=False)

    def predict_with_kg(self, query: str, kg_context: str) -> str:
        """
        Predict medicine recommendation with knowledge graph context.

        Args:
            query: Patient query with symptoms
            kg_context: Relevant medical knowledge from KG

        Returns:
            Medicine recommendation
        """
        prompt = f"""基于医学知识库推荐中药：

医学知识：
{kg_context}

问题：{query}

基于以上医学知识，推荐中药："""

        if self.model is None:
            # Mock response for demonstration
            return self._mock_prediction(query, use_kg=True, context=kg_context)

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the generated part
            recommendation = response.split("推荐中药：")[-1].strip()

            return recommendation if recommendation else "无法生成推荐"

        except Exception as e:
            print(f"Error during prediction: {e}")
            return self._mock_prediction(query, use_kg=True, context=kg_context)

    def _mock_prediction(
        self,
        query: str,
        use_kg: bool = False,
        context: str = ""
    ) -> str:
        """
        Mock prediction for demonstration when model is not available.

        Args:
            query: Patient query
            use_kg: Whether KG context was used
            context: KG context if available

        Returns:
            Mock medicine recommendation
        """
        # Simple rule-based mock for demonstration
        symptom_drug_map = {
            '头痛': '银翘解毒片',
            '发热': '连花清瘟胶囊',
            '咳嗽': '川贝枇杷膏',
            '流鼻涕': '板蓝根颗粒',
            '腹痛': '藿香正气丸',
            '腹泻': '补中益气丸',
            '失眠': '安神补脑液',
            '心悸': '天王补心丹',
            '口苦': '龙胆泻肝丸',
            '口干': '玄麦甘桔颗粒'
        }

        recommendations = []
        for symptom, drug in symptom_drug_map.items():
            if symptom in query:
                recommendations.append(drug)

        if not recommendations:
            recommendations = ['六味地黄丸', '逍遥丸']

        result = "、".join(recommendations[:2])

        if use_kg and context:
            result += " (基于知识图谱推荐)"
        else:
            result += " (基于LLM直接推荐)"

        return result


class LightweightLLMPredictor(MedicineLLMPredictor):
    """
    Extra lightweight predictor using even smaller models.
    """

    def __init__(self, device: str = "cpu"):
        """
        Initialize with a very lightweight model.

        Args:
            device: Device to run on
        """
        # Use smaller Chinese models
        super().__init__(
            model_name="uer/gpt2-chinese-cluecorpussmall",
            device=device,
            max_length=128
        )


if __name__ == "__main__":
    # Test the predictor
    print("Testing Medicine LLM Predictor...")

    predictor = MedicineLLMPredictor()

    # Test without KG
    query1 = "患者出现头痛发热症状。请推荐合适的中药。"
    print(f"\n查询: {query1}")
    print("不使用知识图谱:")
    result1 = predictor.predict_without_kg(query1)
    print(f"推荐: {result1}")

    # Test with KG
    kg_context = """
疾病名称: 感冒
描述: 感冒是由病毒引起的上呼吸道感染，常见症状包括发热、头痛、咳嗽等。
治疗方法: 可使用银翘解毒片、连花清瘟胶囊等中成药。
"""
    print("\n使用知识图谱:")
    result2 = predictor.predict_with_kg(query1, kg_context)
    print(f"推荐: {result2}")
