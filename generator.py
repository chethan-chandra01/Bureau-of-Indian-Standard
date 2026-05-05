"""
src/generator.py
----------------
Phase 3: The Generator (Ollama)
Optimized for CPU: Truncates context size to prevent extreme latency,
and uses Ollama's native JSON format parameter to prevent parsing errors.
"""

import json
import logging
import ollama

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an Expert AI Compliance Officer for the Bureau of Indian Standards (BIS).
Your task is to match a user's product description to the provided candidate standards.

RULES OF ENGAGEMENT:
1. ZERO HALLUCINATION: You may ONLY recommend standards that are explicitly listed in the CONTEXT below.
2. STRICT MATCHING: If a standard in the context does not apply to the product, discard it.

EXPECTED JSON SCHEMA:
{
  "retrieved_standards": ["IS 1234: 5678"],
  "rationale": "A brief explanation of why these specific standards apply."
}
"""

def build_context(retrieved_results: list) -> str:
    """Formats the retrieved FAISS chunks. Truncated to save CPU compute time."""
    context_str = "CONTEXT (Candidate Standards):\n\n"
    for i, res in enumerate(retrieved_results, 1):
        context_str += f"--- STANDARD {i} ---\n"
        context_str += f"IS Number: {res['is_number']}\n"
        context_str += f"Title: {res['title']}\n"
        context_str += f"Scope: {res['scope']}\n"
        # Truncate full text to 800 chars so we don't choke the CPU
        full_text_snippet = res.get('full_text', '')[:800].replace('\n', ' ')
        context_str += f"Text Snippet: {full_text_snippet}...\n\n"
    return context_str

def generate_answer(query: str, retrieved_results: list, model: str = "llama3") -> dict:
    """
    Passes the query and retrieved context to the LLM.
    Returns a dictionary matching the required JSON schema.
    """
    context = build_context(retrieved_results)
    user_prompt = f"USER PRODUCT QUERY: {query}\n\n{context}"

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            format='json', # <--- MAGIC FIX: Forces Ollama to output valid JSON only
            options={"temperature": 0.0}
        )
        
        raw_output = response['message']['content'].strip()
        parsed_output = json.loads(raw_output)
        return parsed_output

    except json.JSONDecodeError:
        logger.error("LLM failed to return valid JSON despite JSON mode.")
        return {
            "retrieved_standards": [r["is_number"] for r in retrieved_results],
            "rationale": "JSON Parse Error. Defaulting to raw retriever output."
        }
    except Exception as e:
        logger.error(f"Ollama generation failed: {e}")
        return {
            "retrieved_standards": [r["is_number"] for r in retrieved_results],
            "rationale": "LLM generation failed."
        }