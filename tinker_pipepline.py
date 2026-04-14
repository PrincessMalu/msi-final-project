"""
MSI Project: Tinker API Pipeline
Testing Social Reasoning in LLMs: A Multi-Dimensional Benchmark for Theory of Mind and Pragmatics
Authors: Kiera McCormick & Malavika Nair

Setup:
    pip install openai python-dotenv

Add to your .env file:
    TINKER_API_KEY=your_actual_key_here

This pipeline:
1. Loads scenarios from a JSON file
2. Queries LLMs via Tinker's OpenAI-compatible API (standard + CoT prompting conditions)
3. Injects per-scenario few-shot examples (standard vs CoT variants) into prompts
4. Scores responses via keyword matching (answer tag only) or LLM grader for open-ended items
5. Flags ambiguous responses for manual review
6. Records scenario subtype and difficulty for binned analysis
"""

import json
import os
import time
import csv
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # loads TINKER_API_KEY from your .env file


# ── Configuration ──────────────────────────────────────────────────────────────

TINKER_API_KEY = os.environ.get("TINKER_API_KEY")
if not TINKER_API_KEY:
    raise EnvironmentError("TINKER_API_KEY not set. Add it to your .env file.")

TINKER_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"

# ── LLM Grader configuration ──────────────────────────────────────────────────
# Qwen3-32B is used to grade open-ended scenarios (scoring_method: "llm_grader").
# It is excluded from MODELS / ALL_MODELS so it never appears in experimental results.
GRADER_MODEL = "Qwen/Qwen3-32B"
GRADER_MODEL_ID = GRADER_MODEL  # alias used in make_grader_client()

# ── POC: single model for midway report ───────────────────────────────────────
# Only querying one model for now as a proof of concept.
# To run the full benchmark, set MODELS = ALL_MODELS below.

MODELS = [
    {"id": "Qwen/Qwen3-8B", "label": "Qwen3-8B", "scale": "8B", "arch": "Dense", "regime": "Hybrid"},
]

# ALL_MODELS — swap in for full data collection during Weeks 3-4
# ALL_MODELS = [
#     # Instruction-tuned (fast inference, no chain-of-thought)
#     {"id": "Qwen/Qwen3-4B-Instruct-2507",       "label": "Qwen3-4B-Instruct",      "scale": "4B",   "arch": "Dense", "regime": "Instruction"},
#     {"id": "Qwen/Qwen3-30B-A3B-Instruct-2507",  "label": "Qwen3-30B-Instruct",     "scale": "30B",  "arch": "MoE",   "regime": "Instruction"},
#     {"id": "Qwen/Qwen3-235B-A22B-Instruct-2507","label": "Qwen3-235B-Instruct",    "scale": "235B", "arch": "MoE",   "regime": "Instruction"},
#     {"id": "meta-llama/Llama-3.1-8B-Instruct",  "label": "Llama-3.1-8B-Instruct",  "scale": "8B",   "arch": "Dense", "regime": "Instruction"},
#     {"id": "meta-llama/Llama-3.3-70B-Instruct", "label": "Llama-3.3-70B-Instruct", "scale": "70B",  "arch": "Dense", "regime": "Instruction"},
#     # Hybrid (thinking + non-thinking modes)
#     {"id": "Qwen/Qwen3-8B",                     "label": "Qwen3-8B",               "scale": "8B",   "arch": "Dense", "regime": "Hybrid"},
#     # Qwen3-32B is reserved as the LLM grader — excluded from experimental models.
    # {"id": "Qwen/Qwen3-32B",                    "label": "Qwen3-32B",              "scale": "32B",  "arch": "Dense", "regime": "Hybrid"},
#     {"id": "Qwen/Qwen3-30B-A3B",                "label": "Qwen3-30B-A3B",          "scale": "30B",  "arch": "MoE",   "regime": "Hybrid"},
#     {"id": "deepseek-ai/DeepSeek-V3.1",         "label": "DeepSeek-V3.1",          "scale": "671B", "arch": "MoE",   "regime": "Hybrid"},
#     # Reasoning-specialized (always uses chain-of-thought internally)
#     {"id": "openai/gpt-oss-20b",                "label": "GPT-OSS-20B",            "scale": "20B",  "arch": "MoE",   "regime": "Reasoning"},
#     {"id": "openai/gpt-oss-120b",               "label": "GPT-OSS-120B",           "scale": "120B", "arch": "MoE",   "regime": "Reasoning"},
#     {"id": "moonshotai/Kimi-K2-Thinking",       "label": "Kimi-K2-Thinking",       "scale": "671B", "arch": "MoE",   "regime": "Reasoning"},
#     # Base (no instruction tuning — for research comparison)
#     {"id": "Qwen/Qwen3-8B-Base",                "label": "Qwen3-8B-Base",          "scale": "8B",   "arch": "Dense", "regime": "Base"},
# ]

# Output directory
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Prompt templates ───────────────────────────────────────────────────────────

STANDARD_SYSTEM_PROMPT = """You are being tested on social reasoning tasks.
Read the scenario carefully and answer the question.
Give a clear, concise answer. End your response with your final answer on a new line
in the format: ANSWER: <your answer>"""

COT_SYSTEM_PROMPT = """You are being tested on social reasoning tasks.
Read the scenario carefully. Before answering, reason step-by-step about:
1. What each person knows or believes
2. What information is available or hidden
3. What the most reasonable conclusion is

Then provide your final answer on a new line in the format: ANSWER: <your answer>"""


def _format_few_shot_examples(examples: list[dict]) -> str:
    """
    Format a list of few-shot example dicts into a prompt block.
    Each example has 'scenario', 'question', and 'answer' keys.
    """
    blocks = []
    for ex in examples:
        blocks.append(
            f"Example:\n{ex['scenario']}\n\nQuestion: {ex['question']}\n{ex['answer']}"
        )
    return "\n\n---\n\n".join(blocks) + "\n\n---\n\n"


def build_prompt(scenario: dict, use_cot: bool = False) -> tuple[str, str]:
    """
    Returns (system_prompt, user_prompt) for a scenario.

    Few-shot examples are drawn from scenario['few_shot_examples'], which is a dict
    with 'standard' and 'cot' keys (list of example dicts each). The correct variant
    is selected based on use_cot:
      - standard: examples contain only the bare ANSWER: line (no reasoning chain)
      - cot:      examples contain full step-by-step reasoning + ANSWER: line

    If few_shot_examples is absent or empty the prompt is returned without examples.
    """
    system = COT_SYSTEM_PROMPT if use_cot else STANDARD_SYSTEM_PROMPT

    # Build few-shot block from the correct variant
    few_shot_block = ""
    fse = scenario.get("few_shot_examples")
    if isinstance(fse, dict):
        variant_key = "cot" if use_cot else "standard"
        examples = fse.get(variant_key, [])
        if examples:
            few_shot_block = _format_few_shot_examples(examples)

    user = f"{few_shot_block}{scenario['scenario_text']}\n\nQuestion: {scenario['question']}"
    return system, user


# ── API query ──────────────────────────────────────────────────────────────────

def make_client() -> OpenAI:
    """Create an OpenAI client pointed at the Tinker inference endpoint."""
    return OpenAI(base_url=TINKER_BASE_URL, api_key=TINKER_API_KEY)


def make_grader_client() -> OpenAI:
    """Return a Tinker client for the Qwen3-32B grader."""
    return OpenAI(base_url=TINKER_BASE_URL, api_key=TINKER_API_KEY)


def query_model(client: OpenAI, model_id: str, system_prompt: str, user_prompt: str,
                max_tokens: int = 512,
                max_retries: int = 3, retry_delay: float = 5.0) -> dict:
    """
    Query a single model via Tinker's OpenAI-compatible API with retry logic.
    Returns a dict with 'text', 'tokens_used', and 'error' fields.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.0,   # deterministic outputs for evaluation
                max_tokens=max_tokens,
            )
            return {
                "text": response.choices[0].message.content,
                "tokens_used": response.usage.total_tokens if response.usage else None,
                "error": None,
            }
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    ⚠️  Attempt {attempt+1} failed for {model_id}: {e}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                return {"text": None, "tokens_used": None, "error": str(e)}


# ── Response parsing ───────────────────────────────────────────────────────────

def extract_answer(response_text: str) -> str | None:
    """Extract the final answer from the model's response."""
    if not response_text:
        return None
    for line in reversed(response_text.strip().splitlines()):
        if line.strip().upper().startswith("ANSWER:"):
            return line.split(":", 1)[1].strip()
    return None  # No structured answer found → will be flagged for manual review


def keyword_score(extracted_answer: str | None, correct_answer: str,
                  keywords: list[str]) -> dict:
    """
    Score a closed-form answer using keyword matching against the extracted answer only.

    IMPORTANT: This function always operates on `extracted_answer` (the text after
    'ANSWER:'), never on the full response. This is intentional — scenarios with
    answer_scope='answer_tag_only' (e.g. tom_014) require this scoped matching to
    avoid false positives from distractors that appear in the reasoning chain.

    Returns dict with 'correct' (bool), 'method' (str), and 'needs_review' (bool).
    """
    if extracted_answer is None:
        return {"correct": None, "method": "no_answer", "needs_review": True}

    answer_lower = extracted_answer.lower()
    correct_lower = correct_answer.lower()

    # Exact match
    if answer_lower == correct_lower:
        return {"correct": True, "method": "exact_match", "needs_review": False}

    # Keyword match
    for kw in keywords:
        if kw.lower() in answer_lower:
            is_correct = kw.lower() in correct_lower
            return {"correct": is_correct, "method": "keyword_match", "needs_review": False}

    # Ambiguous — flag for manual review
    return {"correct": None, "method": "ambiguous", "needs_review": True}


def llm_grade(response_text: str | None, extracted_answer: str | None,
              scenario: dict) -> dict:
    """
    Score an open-ended response using an LLM grader (for scenarios with
    scoring_method='llm_grader'). Uses the scenario's grading_rubric field
    to construct the grading prompt.

    Returns dict with 'correct' (bool|None), 'method' (str), 'needs_review' (bool),
    and 'grader_reasoning' (str).
    """
    if not response_text or not extracted_answer:
        return {
            "correct": None, "method": "llm_grader_no_answer",
            "needs_review": True, "grader_reasoning": None,
        }

    rubric    = scenario.get("grading_rubric", "Award credit if the answer is logically sound and addresses the scenario correctly.")
    grader_prompt = f"""You are an expert grader for a social reasoning benchmark.

Scenario: {scenario['scenario_text']}
Question: {scenario['question']}
Correct answer (reference): {scenario['correct_answer']}
Grading rubric: {rubric}

Model's extracted answer (after ANSWER: tag):
{extracted_answer}

Based on the rubric, is this answer correct? Reply with exactly one of:
GRADE: correct
GRADE: incorrect
GRADE: partial

Then on a new line, give a one-sentence explanation.
"""
    try:
        grader_client = make_grader_client()
        resp = grader_client.chat.completions.create(
            model=GRADER_MODEL,
            messages=[{"role": "user", "content": grader_prompt}],
            temperature=0.0,
            max_tokens=128,
        )
        grader_text = resp.choices[0].message.content.strip()
        grade_line  = next((l for l in grader_text.splitlines()
                            if l.strip().upper().startswith("GRADE:")), None)
        if grade_line:
            grade = grade_line.split(":", 1)[1].strip().lower()
            correct = True if grade == "correct" else (None if grade == "partial" else False)
            needs_review = grade == "partial"
        else:
            correct, needs_review = None, True

        return {
            "correct":          correct,
            "method":           "llm_grader",
            "needs_review":     needs_review,
            "grader_reasoning": grader_text,
        }
    except Exception as e:
        return {
            "correct": None, "method": "llm_grader_error",
            "needs_review": True, "grader_reasoning": str(e),
        }


def score_cot(response_text: str | None) -> dict:
    """
    Placeholder for CoT quality scoring on 4 dimensions (0-2 each, total 0-8).
    All CoT responses are flagged for manual scoring during Week 5.

    Dimensions:
      - mental_state_reasoning:    Does the model attribute beliefs/knowledge correctly?
      - alternative_consideration: Does it consider alternative interpretations?
      - step_by_step_structure:    Is reasoning sequential and organized?
      - logical_coherence:         Is the chain of reasoning internally consistent?
    """
    return {
        "mental_state_reasoning":    None,
        "alternative_consideration": None,
        "step_by_step_structure":    None,
        "logical_coherence":         None,
        "total_cot_score":           None,
        "needs_cot_review":          response_text is not None,
    }


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(scenarios_path: str = "scenarios.json"):
    """
    Main data collection pipeline.
    Loads scenarios, queries all models (standard + CoT), stores results.
    """
    with open(scenarios_path) as f:
        scenarios = json.load(f)
    print(f" Loaded {len(scenarios)} scenarios from {scenarios_path}")

    client = make_client()
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_queries = len(scenarios) * len(MODELS) * 2  # standard + CoT
    completed = 0

    for scenario in scenarios:
        scenario_id     = scenario["id"]
        scenario_type   = scenario["type"]
        scenario_subtype = scenario.get("subtype", "")
        difficulty      = scenario.get("difficulty", "")
        correct_answer  = scenario["correct_answer"]
        keywords        = scenario.get("keywords", [correct_answer])
        scoring_method  = scenario.get("scoring_method", "keyword")  # "keyword" | "llm_grader"

        for model in MODELS:
            model_id = model["id"]

            for use_cot in [False, True]:
                condition = "cot" if use_cot else "standard"
                system_prompt, user_prompt = build_prompt(scenario, use_cot=use_cot)

                print(f"  [{completed+1}/{total_queries}] {model['label']} | {scenario_id} | {condition}")

                # CoT responses need more space for pragmatic/social reasoning chains
                tokens = 1024 if use_cot else 512
                response = query_model(client, model_id, system_prompt, user_prompt,
                                       max_tokens=tokens)
                response_text    = response["text"]
                extracted_answer = extract_answer(response_text)

                # ── Scoring: route by scoring_method ──────────────────────────
                if scoring_method == "llm_grader":
                    scoring = llm_grade(response_text, extracted_answer, scenario)
                else:
                    scoring = keyword_score(extracted_answer, correct_answer, keywords)

                cot_scoring = score_cot(response_text) if use_cot else {}

                result = {
                    # Identifiers
                    "scenario_id":      scenario_id,
                    "scenario_type":    scenario_type,
                    "scenario_subtype": scenario_subtype,
                    "difficulty":       difficulty,
                    "model_id":         model_id,
                    "model_label":      model["label"],
                    "model_scale":      model["scale"],
                    "model_arch":       model["arch"],
                    "model_regime":     model["regime"],
                    "condition":        condition,
                    # Prompts & response
                    "system_prompt":    system_prompt,
                    "user_prompt":      user_prompt,
                    "response_text":    response_text,
                    "api_error":        response["error"],
                    "tokens_used":      response["tokens_used"],
                    # Scoring
                    "correct_answer":   correct_answer,
                    "extracted_answer": extracted_answer,
                    "is_correct":       scoring["correct"],
                    "score_method":     scoring["method"],
                    "needs_review":     scoring["needs_review"],
                    "grader_reasoning": scoring.get("grader_reasoning"),  # None for keyword items
                    # CoT scoring (filled in during manual review, Week 5)
                    **cot_scoring,
                    # Metadata
                    "timestamp": datetime.now().isoformat(),
                }
                results.append(result)
                completed += 1
                time.sleep(0.5)  # gentle rate limiting

        # Save incrementally after each scenario (safe against crashes)
        _save_results(results, timestamp)

    print(f"\n Pipeline complete. {completed} queries run.")
    print(f"📁 Results saved to {OUTPUT_DIR}/results_{timestamp}.json and .csv")
    _print_summary(results)
    return results


# ── Storage helpers ────────────────────────────────────────────────────────────

def _save_results(results: list[dict], timestamp: str):
    """Save results as both JSON and CSV."""
    json_path = OUTPUT_DIR / f"results_{timestamp}.json"
    csv_path  = OUTPUT_DIR / f"results_{timestamp}.csv"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    if results:
        # Collect all keys across all rows (standard rows lack CoT fields)
        all_keys = list(dict.fromkeys(k for r in results for k in r.keys()))
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)


def _print_summary(results: list[dict]):
    """Print a quick summary of results and flagged items."""
    total        = len(results)
    needs_review = sum(1 for r in results if r.get("needs_review"))
    errors       = sum(1 for r in results if r.get("api_error"))

    print(f"\n── Summary ──────────────────────────────")
    print(f"  Total responses:    {total}")
    print(f"  Flagged for review: {needs_review} ({100*needs_review//total if total else 0}%)")
    print(f"  API errors:         {errors}")

    from collections import defaultdict

    # Per-model accuracy (auto-scored only)
    model_scores = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        if r.get("is_correct") is not None:
            model_scores[r["model_label"]]["total"] += 1
            if r["is_correct"]:
                model_scores[r["model_label"]]["correct"] += 1

    print(f"\n── Per-model accuracy (auto-scored only) ─")
    for label, s in sorted(model_scores.items()):
        pct = 100 * s["correct"] / s["total"] if s["total"] else 0
        print(f"  {label:<35} {s['correct']}/{s['total']} ({pct:.1f}%)")

    # Per-difficulty accuracy — key for the binned analysis the professor requested
    diff_scores = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        if r.get("is_correct") is not None and r.get("difficulty"):
            diff_scores[r["difficulty"]]["total"] += 1
            if r["is_correct"]:
                diff_scores[r["difficulty"]]["correct"] += 1

    if diff_scores:
        print(f"\n── Accuracy by difficulty tier ───────────")
        for tier in ["basic", "intermediate", "advanced"]:
            s = diff_scores.get(tier)
            if s and s["total"]:
                pct = 100 * s["correct"] / s["total"]
                print(f"  {tier:<15} {s['correct']}/{s['total']} ({pct:.1f}%)")

    # Per-subtype accuracy
    subtype_scores = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        if r.get("is_correct") is not None and r.get("scenario_subtype"):
            subtype_scores[r["scenario_subtype"]]["total"] += 1
            if r["is_correct"]:
                subtype_scores[r["scenario_subtype"]]["correct"] += 1

    if subtype_scores:
        print(f"\n── Accuracy by scenario subtype ──────────")
        for subtype, s in sorted(subtype_scores.items()):
            pct = 100 * s["correct"] / s["total"] if s["total"] else 0
            print(f"  {subtype:<40} {s['correct']}/{s['total']} ({pct:.1f}%)")


# ── Manual review export ───────────────────────────────────────────────────────

def export_for_manual_review(results_path: str):
    """Export flagged responses to a separate CSV for manual review (Week 5)."""
    with open(results_path) as f:
        results = json.load(f)

    flagged     = [r for r in results if r.get("needs_review") or r.get("needs_cot_review")]
    review_path = results_path.replace(".json", "_manual_review.csv")

    if flagged:
        cols = ["scenario_id", "scenario_type", "scenario_subtype", "difficulty",
                "model_label", "condition",
                "user_prompt", "response_text", "extracted_answer",
                "correct_answer", "score_method", "grader_reasoning",
                "needs_review", "needs_cot_review",
                "mental_state_reasoning", "alternative_consideration",
                "step_by_step_structure", "logical_coherence", "total_cot_score"]
        with open(review_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(flagged)
        print(f"📋 {len(flagged)} items exported for manual review → {review_path}")
    else:
        print(" No items flagged for manual review.")


# ── Example scenario schema ────────────────────────────────────────────────────

EXAMPLE_SCENARIOS = [
    {
        "id": "tom_001",
        "type": "theory_of_mind",
        "scenario_text": (
            "Sally puts her marble in a basket and leaves the room. "
            "Anne moves the marble to a box while Sally is away. "
            "Sally comes back into the room."
        ),
        "question": "Where will Sally look for her marble?",
        "correct_answer": "basket",
        "keywords": ["basket"],
        "reasoning_type": "false_belief",
        "difficulty": "basic",
    },
    {
        "id": "prag_001",
        "type": "pragmatics",
        "scenario_text": (
            "Alex asks: 'Can you pass the salt?' "
            "There is salt right in front of Jordan on the table."
        ),
        "question": "What should Jordan do?",
        "correct_answer": "pass the salt",
        "keywords": ["pass", "salt", "hand"],
        "reasoning_type": "indirect_speech_act",
        "difficulty": "basic",
    },
    {
        "id": "cot_001",
        "type": "chain_of_thought",
        "scenario_text": (
            "John told Mary he would meet her at the coffee shop at 3pm. "
            "Mary arrives at 3pm but John is not there. "
            "John texts: 'I'm running a bit late.'"
        ),
        "question": "What does Mary most likely infer about when John will arrive?",
        "correct_answer": "soon / shortly after 3pm",
        "keywords": ["soon", "shortly", "few minutes", "late"],
        "reasoning_type": "pragmatic_inference",
        "difficulty": "intermediate",
    },
]


def create_example_scenarios_file():
    """Write example scenario file to disk for testing the pipeline."""
    with open("scenarios.json", "w") as f:
        json.dump(EXAMPLE_SCENARIOS, f, indent=2)
    print("Example scenarios.json created. Replace with your full 40 scenarios.")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MSI Project Tinker API Pipeline")
    parser.add_argument("--scenarios", default="scenarios.json",
                        help="Path to scenarios JSON file")
    parser.add_argument("--create-example", action="store_true",
                        help="Create an example scenarios.json file")
    parser.add_argument("--export-review", type=str, default=None,
                        help="Path to results JSON to export flagged items for manual review")
    args = parser.parse_args()

    if args.create_example:
        create_example_scenarios_file()
    elif args.export_review:
        export_for_manual_review(args.export_review)
    else:
        run_pipeline(scenarios_path=args.scenarios)