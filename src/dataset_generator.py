"""Generates ARC-specific fine-tuning datasets for Laya in JSONL format."""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def extract_arc2_decision_examples(
    challenges_file: Path,
    solutions_file: Path,
    limit: int = 500,
) -> List[Dict[str, Any]]:
    """Extract transformation rule intuition examples from ARC-AGI-2 tasks."""
    if not challenges_file.exists() or not solutions_file.exists():
        return []

    with open(challenges_file, "r") as f:
        challenges = json.load(f)
    with open(solutions_file, "r") as f:
        solutions = json.load(f)

    examples = []
    task_keys = list(challenges.keys())[:limit]

    for task_id in task_keys:
        task = challenges[task_id]
        train_pairs = task.get("train", [])
        test_pairs = task.get("test", [])
        test_sols = solutions.get(task_id, [])

        if not train_pairs or not test_pairs:
            continue

        # Summarize training transformations
        size_preserved = all(
            len(p["input"]) == len(p["output"]) and len(p["input"][0]) == len(p["output"][0])
            for p in train_pairs
        )
        color_count_input = len(set(c for p in train_pairs for r in p["input"] for c in r))
        color_count_output = len(set(c for p in train_pairs for r in p["output"] for c in r))
        new_colors_introduced = color_count_output > color_count_input

        # Build state description
        state_obj = {
            "task_id": task_id,
            "benchmark": "ARC-AGI-2",
            "num_demonstrations": len(train_pairs),
            "input_dim": f"{len(train_pairs[0]['input'])}x{len(train_pairs[0]['input'][0])}",
            "size_invariant": size_preserved,
            "unique_input_colors": color_count_input,
            "unique_output_colors": color_count_output,
        }

        # Questions for System 1 heuristic screening
        questions = {
            "transformation_type": {
                "type": "choice",
                "instructions": "Classify the primary transformation mode needed to solve this puzzle",
                "criteria": {
                    "geometry": "Geometric reflection, rotation, translation, or tiling",
                    "flood_fill": "Coloring enclosed regions or connected components",
                    "counting_sorting": "Mapping objects based on frequency, size, or count",
                    "pattern_extrapolation": "Extending recurring sequences or completing missing sections",
                },
            },
            "output_size_mode": {
                "type": "choice",
                "instructions": "Predict output grid dimension relationship to input grid",
                "criteria": {
                    "same_size": "Output grid dimensions exactly match input grid",
                    "compressed": "Output grid is strictly smaller (cropping or abstraction)",
                    "expanded": "Output grid is larger (tiling or scaling)",
                },
            },
            "requires_deep_search": {
                "type": "noul",
                "instructions": "Solving this task requires multi-step combinatorial program search",
            },
        }

        # Determine gold labels based on ground truth data
        size_label = "same_size" if size_preserved else ("compressed" if color_count_output <= 2 else "expanded")
        
        gold = {
            "output_size_mode": {
                "type": "choice",
                "label": size_label,
                "probabilities": {
                    "same_size": 0.85 if size_preserved else 0.08,
                    "compressed": 0.08 if size_preserved else 0.82,
                    "expanded": 0.07 if size_preserved else 0.10,
                },
                "confidence": 0.85,
            },
            "transformation_type": {
                "type": "choice",
                "label": "geometry" if size_preserved else "pattern_extrapolation",
                "probabilities": {
                    "geometry": 0.70 if size_preserved else 0.10,
                    "flood_fill": 0.10,
                    "counting_sorting": 0.10,
                    "pattern_extrapolation": 0.10 if size_preserved else 0.70,
                },
                "confidence": 0.70,
            },
            "requires_deep_search": {
                "type": "noul",
                "label": "true" if new_colors_introduced else "false",
                "probabilities": {
                    "true": 0.80 if new_colors_introduced else 0.20,
                    "false": 0.20 if new_colors_introduced else 0.80,
                },
                "confidence": 0.80,
            },
        }

        examples.append({
            "workflow": "arc2_heuristic_screening",
            "split": "train",
            "state": json.dumps(state_obj),
            "questions": json.dumps(questions),
            "gold": json.dumps(gold),
            "factors": json.dumps({"size_preserved": size_preserved, "benchmark": "ARC-AGI-2"}),
            "n_questions": 3,
        })

    return examples


def extract_arc3_navigation_examples(
    num_synthetic_episodes: int = 400,
) -> List[Dict[str, Any]]:
    """Synthesize navigation & impasse decision traces for ARC-AGI-3 games."""
    examples = []
    actions = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION7"]
    
    for i in range(num_synthetic_episodes):
        step = random.randint(1, 40)
        is_stuck = random.random() < 0.25
        is_near_goal = random.random() < 0.30
        
        target_action = "ACTION5" if is_near_goal else ("ACTION7" if is_stuck else random.choice(["ACTION1", "ACTION2", "ACTION3", "ACTION4"]))
        
        state_obj = {
            "game_id": f"env_sample_{i % 25:02d}",
            "benchmark": "ARC-AGI-3",
            "action_step": step,
            "recent_actions": [random.choice(actions) for _ in range(min(step, 4))],
            "agent_detected": True,
            "is_stagnant": is_stuck,
            "goal_visible": is_near_goal,
        }

        questions = {
            "recommended_action": {
                "type": "choice",
                "instructions": "Select next game action based on current local situation",
                "criteria": {
                    "ACTION1": "Move UP",
                    "ACTION2": "Move DOWN",
                    "ACTION3": "Move LEFT",
                    "ACTION4": "Move RIGHT",
                    "ACTION5": "Interact / Trigger",
                    "ACTION7": "Undo last action",
                },
            },
            "impasse_detected": {
                "type": "noul",
                "instructions": "The agent is currently blocked or oscillating without progress",
            },
            "escalate_to_system2": {
                "type": "score",
                "instructions": "Need for heavy analytical reasoning / tree search",
                "criteria": [
                    "Low (routine navigation)",
                    "Medium (evaluating alternative branches)",
                    "High (deadlock / new unseen mechanic)",
                ],
            },
        }

        gold = {
            "recommended_action": {
                "type": "choice",
                "label": target_action,
                "probabilities": {
                    act: (0.75 if act == target_action else 0.05) for act in actions
                },
                "confidence": 0.75,
            },
            "impasse_detected": {
                "type": "noul",
                "label": "true" if is_stuck else "false",
                "probabilities": {
                    "true": 0.90 if is_stuck else 0.10,
                    "false": 0.10 if is_stuck else 0.90,
                },
                "confidence": 0.90,
            },
            "escalate_to_system2": {
                "type": "score",
                "label": "2" if is_stuck else ("1" if is_near_goal else "0"),
                "probabilities": {
                    "0": 0.10 if is_stuck else (0.20 if is_near_goal else 0.70),
                    "1": 0.20 if is_stuck else (0.70 if is_near_goal else 0.20),
                    "2": 0.70 if is_stuck else (0.10 if is_near_goal else 0.10),
                },
                "confidence": 0.70,
            },
        }

        examples.append({
            "workflow": "arc3_agent_gating",
            "split": "train",
            "state": json.dumps(state_obj),
            "questions": json.dumps(questions),
            "gold": json.dumps(gold),
            "factors": json.dumps({"is_stuck": is_stuck, "is_near_goal": is_near_goal}),
            "n_questions": 3,
        })

    return examples


def build_and_save_datasets(output_dir: Path):
    """Build train and holdout datasets and save to JSONL."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    project_root = Path(__file__).resolve().parent.parent
    arc2_dir = project_root / "data" / "arc-agi-2"
    challenges = arc2_dir / "arc-agi_training_challenges.json"
    solutions = arc2_dir / "arc-agi_training_solutions.json"

    print("Extracting ARC-AGI-2 examples...")
    arc2_examples = extract_arc2_decision_examples(challenges, solutions, limit=500)
    print(f"-> Generated {len(arc2_examples)} ARC-AGI-2 screening examples.")

    print("Generating ARC-AGI-3 navigation examples...")
    arc3_examples = extract_arc3_navigation_examples(num_synthetic_episodes=500)
    print(f"-> Generated {len(arc3_examples)} ARC-AGI-3 navigation examples.")

    all_examples = arc2_examples + arc3_examples
    random.seed(42)
    random.shuffle(all_examples)

    # 80/20 Train / Holdout split
    split_idx = int(len(all_examples) * 0.8)
    train_data = all_examples[:split_idx]
    holdout_data = all_examples[split_idx:]

    for ex in train_data:
        ex["split"] = "train"
    for ex in holdout_data:
        ex["split"] = "holdout"

    train_path = output_dir / "train.jsonl"
    holdout_path = output_dir / "holdout.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    with open(holdout_path, "w", encoding="utf-8") as f:
        for item in holdout_data:
            f.write(json.dumps(item) + "\n")

    print(f"\n✅ Successfully created datasets in {output_dir}:")
    print(f"   • Train set:   {len(train_data)} records ({train_path.name})")
    print(f"   • Holdout set: {len(holdout_data)} records ({holdout_path.name})")


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "data" / "laya_finetune"
    build_and_save_datasets(out)
