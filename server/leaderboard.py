import time
from typing import Dict, List, Optional


# ==============================================================
# Leaderboard — module-level singleton
# ==============================================================

_board: Dict[str, dict] = {}


def _safe_score(score: float) -> float:
    """Ensure score is strictly between 0 and 1."""
    return round(max(0.01, min(float(score), 0.99)), 4)


def submit(
    model_name: str,
    overall_score: float,
    tasks_solved: int,
    skill_scores: Optional[Dict[str, float]] = None,
):
    overall_score = _safe_score(overall_score)
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

        new_avg = _safe_score(
            (prev_avg * prev_runs + overall_score) / (prev_runs + 1)
        )

        updated_skills = dict(entry["skill_scores"])
        for bug_type, score in skill_scores.items():
            updated_skills[bug_type] = max(
                updated_skills.get(bug_type, 0.01), score
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
    board = get_leaderboard(limit=len(_board))
    for entry in board:
        if entry["model"] == model_name:
            return entry
    return None


def reset_leaderboard():
    """Wipe all leaderboard data. Useful for testing."""
    _board.clear()


def formatted_leaderboard() -> str:
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