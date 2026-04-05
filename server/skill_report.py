from typing import Dict, List


# Maps bug_type tag → human readable category name
BUG_TYPE_LABELS = {
    "syntax":      "Syntax Errors",
    "type":        "Type Errors",
    "logic":       "Logic Errors",
    "off_by_one":  "Off-by-One Errors",
    "edge_case":   "Edge Case Handling",
    "concurrency": "Concurrency & Thread Safety",
    "performance": "Performance & Memory Efficiency",
}

# Thresholds for rating labels
RATING_THRESHOLDS = [
    (0.90, "Excellent"),
    (0.70, "Good"),
    (0.50, "Needs Improvement"),
    (0.00, "Weak"),
]


def _rating(score: float) -> str:
    for threshold, label in RATING_THRESHOLDS:
        if score >= threshold:
            return label
    return "Weak"


class SkillTracker:
    """
    Tracks per-bug-type scores across multiple episodes.
    One SkillTracker instance lives inside AISEEnv and persists
    across reset() calls so it can accumulate data over a full run.

    Usage:
        tracker = SkillTracker()
        tracker.record("edge_case", 0.8)
        tracker.record("syntax", 1.0)
        report = tracker.report()
    """

    def __init__(self):
        # bug_type → list of scores recorded for that category
        self._scores: Dict[str, List[float]] = {k: [] for k in BUG_TYPE_LABELS}

    def record(self, bug_type: str, score: float):
        """Record the final score for a completed episode."""
        if bug_type in self._scores:
            self._scores[bug_type].append(round(score, 4))

    def reset_all(self):
        """Wipe all recorded scores (call between full benchmark runs)."""
        self._scores = {k: [] for k in BUG_TYPE_LABELS}

    def report(self) -> dict:
        """
        Returns a structured skill report card.

        {
          "summary": {
            "tasks_attempted": int,
            "tasks_solved":    int,
            "overall_score":   float,
            "overall_rating":  str,
          },
          "skills": {
            "syntax": {
              "label":   "Syntax Errors",
              "attempts": int,
              "avg_score": float,
              "rating":  str,
            },
            ...
          },
          "strengths":   [list of category labels],
          "weaknesses":  [list of category labels],
          "verdict":     str,   # one-line natural language summary
        }
        """
        attempted = []
        for bug_type, scores in self._scores.items():
            for s in scores:
                attempted.append((bug_type, s))

        tasks_attempted = len(attempted)
        tasks_solved = sum(1 for _, s in attempted if s >= 0.9)
        overall_score = (
            round(sum(s for _, s in attempted) / tasks_attempted, 4)
            if tasks_attempted > 0 else 0.0
        )

        skills = {}
        strengths = []
        weaknesses = []

        for bug_type, label in BUG_TYPE_LABELS.items():
            scores = self._scores[bug_type]
            if not scores:
                continue
            avg = round(sum(scores) / len(scores), 4)
            rating = _rating(avg)
            skills[bug_type] = {
                "label":    label,
                "attempts": len(scores),
                "avg_score": avg,
                "rating":   rating,
            }
            if avg >= 0.80:
                strengths.append(label)
            elif avg < 0.50:
                weaknesses.append(label)

        verdict = _build_verdict(overall_score, strengths, weaknesses)

        return {
            "summary": {
                "tasks_attempted": tasks_attempted,
                "tasks_solved":    tasks_solved,
                "overall_score":   overall_score,
                "overall_rating":  _rating(overall_score),
            },
            "skills":    skills,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "verdict":   verdict,
        }

    def formatted_report(self) -> str:
        """
        Returns the report card as a human-readable string.
        Useful for logging and the HF Space UI.
        """
        r = self.report()
        lines = []
        lines.append("=" * 52)
        lines.append("  AI SOFTWARE ENGINEER — SKILL REPORT CARD")
        lines.append("=" * 52)

        s = r["summary"]
        lines.append(
            f"  Tasks Attempted : {s['tasks_attempted']}"
        )
        lines.append(
            f"  Tasks Solved    : {s['tasks_solved']}"
        )
        lines.append(
            f"  Overall Score   : {s['overall_score']:.2f}  [{s['overall_rating']}]"
        )
        lines.append("-" * 52)
        lines.append(f"  {'Category':<30} {'Score':>6}  Rating")
        lines.append("-" * 52)

        for bug_type, data in r["skills"].items():
            bar = _bar(data["avg_score"])
            lines.append(
                f"  {data['label']:<30} {data['avg_score']:>5.2f}  {data['rating']}"
            )
            lines.append(f"    {bar}")

        lines.append("-" * 52)

        if r["strengths"]:
            lines.append(f"  Strengths  : {', '.join(r['strengths'])}")
        if r["weaknesses"]:
            lines.append(f"  Weaknesses : {', '.join(r['weaknesses'])}")

        lines.append("")
        lines.append(f"  Verdict: {r['verdict']}")
        lines.append("=" * 52)

        return "\n".join(lines)


# ==============================================================
# Helpers
# ==============================================================

def _bar(score: float, width: int = 20) -> str:
    """Simple ASCII progress bar for the formatted report."""
    filled = round(score * width)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {score:.0%}"


def _build_verdict(overall: float, strengths: List[str], weaknesses: List[str]) -> str:
    if overall >= 0.90:
        return "Outstanding — this model handles all bug categories with high reliability."
    if overall >= 0.70:
        if weaknesses:
            weak_str = " and ".join(weaknesses[:2])
            return f"Strong overall, but struggles with {weak_str}."
        return "Strong across the board with consistent performance."
    if overall >= 0.50:
        if strengths:
            strong_str = " and ".join(strengths[:2])
            return f"Average performance. Most reliable at {strong_str}."
        return "Average performance with inconsistent results across categories."
    return (
        "Below average. The model needs significant improvement "
        "in most bug categories."
    )
