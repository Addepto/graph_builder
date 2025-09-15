"""
Example: Using AutomaticHeaderExtractor with and without an LLM
"""

import logging
from graph_builder.automatic_header_extractor import (
    AutomaticHeaderExtractor,
    HeaderExtractionMethod,
    HeaderExtractionResult,
)

# ---- Optional: Mock LLM client (replace with OpenAI/Azure/etc.) ----
class MockLLMClient:
    def ask(self, prompt: str) -> str:
        # Simplified: returns 0.9 if 'id' or 'name' present, else 0.5
        return {"score": 0.9 if any(k in prompt.lower() for k in ["id", "name"]) else 0.5}


def run_example():
    logging.basicConfig(level=logging.INFO)

    # Path to your test document (CSV/Excel/PDF)
    file_path = "examples/sample.csv"

    # 1) Pure heuristic extraction
    extractor = AutomaticHeaderExtractor(
        method=HeaderExtractionMethod.HYBRID,
        confidence_threshold=0.5,
    )
    result: HeaderExtractionResult = extractor.extract(file_path)
    print("Heuristic headers:", result.headers)

    # 2) LLM-boosted extraction
    llm_client = MockLLMClient()
    extractor_llm = AutomaticHeaderExtractor(
        method=HeaderExtractionMethod.HYBRID,
        confidence_threshold=0.5,
        llm_client=llm_client,
        llm_weight=0.5,
    )
    result_llm: HeaderExtractionResult = extractor_llm.extract(file_path)
    print("LLM-refined headers:", result_llm.headers)


if __name__ == "__main__":
    run_example()
