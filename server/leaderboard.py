import time
from typing import Dict, List, Optional


# ==============================================================
# Leaderboard — module-level singleton
# ==============================================================
# Lives at module level so it persists across multiple AISEEnv
# instances within the same process. When the FastAPI server
# runs, this dict stays alive for the lifetime of the server.
#
# Structure:
#   _board[model_name] = {
#       "model":        str,
#       "best_score":   float,        # highest overall score achieved
#       "avg_score":    float,        # average across all submissions
#       "runs":         int,          # number of full runs submitted
#       "tasks_solved": int,          # total tasks solved (score >= 0.9)
#       "skill_scores": dict,         # bug_type → best score seen
#       "last_updated": float,        # unix timestamp
#   }
# ==============================================================

_board: Dict[str, dict] = {}


def submit(
    model_name: str,
    overall_score: float,
    tasks_solved: int,
    skill_scores: Optional[Dict[str, float]] = None,
):
    """
    Submit a completed run to the leaderboard.

    Args:
        model_name:    identifier for the model (e.g. "Qwen/Qwen2.5-72B")
        overall_score: normalised score strictly between 0 and 1
        tasks_solved:  number of tasks where score >= 0.9
        skill_scores:  optional dict of bug_type → score for this run
    """
    # Ensure score is strictly between 0 and 1 (not 0.0 and not 1.0)
    overall_score = _safe_score(overall_score)
    if overall_score >= 1.0:
        overall_score = 0.99
    elif overall_score <= 0.0:
        overall_score = 0.01
    
    skill_scores = {k: _safe_score(v) for k, v in (skill_scores or {}).items()}

    if model_name not in _board:
        _board[model_name] = {
            "model":        model_name,
            "best_score":   overall_score,
            "avg_score":    overall_score,
            "runs":         1,
            "tasks_solved": tasks_solved,
            "skill_scores": skill_scores,
            "last_updated": time.time(),
        }
    else:
        entry = _board[model_name]
        prev_avg = entry["avg_score"]
        prev_runs = entry["runs"]

        # Running average
        new_avg = round(
            (prev_avg * prev_runs + overall_score) / (prev_runs + 1), 4
        )

        # Update best skill scores per category
        updated_skills = dict(entry["skill_scores"])
        for bug_type, score in skill_scores.items():
            updated_skills[bug_type] = max(
                updated_skills.get(bug_type, 0.01),
                score,
            )

        _board[model_name] = {
            "model":        model_name,
            "best_score":   max(entry["best_score"], overall_score),
            "avg_score":    new_avg,
            "runs":         prev_runs + 1,
            "tasks_solved": max(entry["tasks_solved"], tasks_solved),
            "skill_scores": updated_skills,
            "last_updated": time.time(),
        }


def get_leaderboard(limit: int = 20) -> List[dict]:
    """
    Returns the leaderboard ranked by best_score descending.
    Each entry includes a rank field.

    Args:
        limit: max number of entries to return (default 20)

    Returns:
        [
          {
            "rank":         1,
            "model":        "Qwen/Qwen2.5-72B",
            "best_score":   0.94,
            "avg_score":    0.87,
            "runs":         3,
            "tasks_solved": 6,
            "skill_scores": {"syntax": 1.0, "logic": 0.8, ...},
            "last_updated": "2025-01-01 12:00:00",
          },
          ...
        ]
    """
    sorted_entries = sorted(
        _board.values(),
        key=lambda e: (e["best_score"], e["tasks_solved"]),
        reverse=True,
    )

    result = []
    for rank, entry in enumerate(sorted_entries[:limit], start=1):
        result.append({
            "rank":         rank,
            "model":        entry["model"],
            "best_score":   entry["best_score"],
            "avg_score":    entry["avg_score"],
            "runs":         entry["runs"],
            "tasks_solved": entry["tasks_solved"],
            "skill_scores": entry["skill_scores"],
            "last_updated": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(entry["last_updated"])
            ),
        })

    return result


def get_model_entry(model_name: str) -> Optional[dict]:
    """
    Returns the leaderboard entry for a specific model, or None if not found.
    Includes its current rank.
    """
    board = get_leaderboard(limit=len(_board))
    for entry in board:
        if entry["model"] == model_name:
            return entry
    return None


def reset_leaderboard():
    """Wipe all leaderboard data. Useful for testing."""
    _board.clear()


def formatted_leaderboard() -> str:
    """
    Returns the leaderboard as a human-readable string.
    Useful for logging and CLI output.
    """
    entries = get_leaderboard()
    if not entries:
        return "Leaderboard is empty — no runs submitted yet."

    lines = []
    lines.append("=" * 62)
    lines.append("  AI SOFTWARE ENGINEER BENCHMARK — LEADERBOARD")
    lines.append("=" * 62)
    lines.append(
        f"  {'Rank':<5} {'Model':<32} {'Best':>6} {'Avg':>6} {'Solved':>7}"
    )
    lines.append("-" * 62)

    for e in entries:
        model_short = e["model"][-32:] if len(e["model"]) > 32 else e["model"]
        lines.append(
            f"  {e['rank']:<5} {model_short:<32} "
            f"{e['best_score']:>6.2f} {e['avg_score']:>6.2f} "
            f"{e['tasks_solved']:>5}/7"
        )

    lines.append("=" * 62)
    return "\n".join(lines)

def _safe_score(score: float) -> float:
    return round(max(0.01, min(float(score), 0.99)), 4)