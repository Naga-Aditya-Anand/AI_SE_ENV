import threading

# ==============================================================
# HARD TASK 1 — Defensive programming (original, kept intact)
# ==============================================================
HARD_TASK = {
    "id": "hard_1",
    "bug_type": "edge_case",

    "code": """def get_average_price(api_responses, window_size):
    total = 0
    for i in range(window_size):
        total += api_responses[i]["closing_price"]
    return total / window_size""",

    "description": (
        "Calculate the average closing price for the given window size from a list "
        "of daily stock data dicts. You must handle messy data: missing keys, "
        "string types, and edge cases where the window is invalid."
    ),

    "expected_solution": """def get_average_price(api_responses, window_size):
    valid_prices = []
    for day in api_responses:
        if "closing_price" in day:
            try:
                valid_prices.append(float(day["closing_price"]))
            except (ValueError, TypeError):
                continue
    if not valid_prices or window_size <= 0 or len(valid_prices) < window_size:
        return 0.0
    return sum(valid_prices[:window_size]) / window_size""",

    "hint": "The code doesn't handle missing keys, non-numeric values, or edge cases. Add validation: check if 'closing_price' exists, convert to float safely, and validate window_size before dividing.",

    "test_cases": [
        {"input": ([{"closing_price": 100}, {"closing_price": 110}, {"closing_price": 120}], 2), "output": 105.0},
        {"input": ([{"closing_price": 100}], 5),                                                  "output": 0.0},
        {"input": ([{"date": "Monday"}, {"closing_price": 150}], 1),                              "output": 150.0},
        {"input": ([{"closing_price": "200"}, {"closing_price": 300}], 2),                        "output": 250.0},
        {"input": ([{"closing_price": 100}], 0),                                                  "output": 0.0},
    ]
}


# ==============================================================
# HARD TASK 2 — Concurrency bug (thread-safety)
# ==============================================================
# Grading note: safe_execute runs in a restricted builtins env,
# so we use "structure_rule": "uses_lock" — the grader checks
# whether the submitted code contains threading.Lock usage via AST.
# The test cases verify correctness under sequential simulation.

HARD_TASK_2 = {
    "id": "hard_2",
    "bug_type": "concurrency",

    "code": """def increment_counter(counter, num_increments):
    for _ in range(num_increments):
        counter["value"] += 1
    return counter["value"]""",

    "description": (
        "This function increments a shared counter dict from multiple threads. "
        "It has a race condition — the read-modify-write on counter['value'] is not atomic. "
        "Fix it by adding a threading.Lock so the increment is thread-safe. "
        "The function signature must stay the same. "
        "Import threading at the top of your solution."
    ),

    "expected_solution": """import threading

_lock = threading.Lock()

def increment_counter(counter, num_increments):
    for _ in range(num_increments):
        with _lock:
            counter["value"] += 1
    return counter["value"]""",

    "hint": "Use threading.Lock() to protect the critical section (counter['value'] += 1). Wrap it in a 'with _lock:' block to ensure atomic access.",

    # Test cases verify correctness (sequential simulation).
    # Thread-safety is checked structurally via "structure_rule".
    "test_cases": [
        {"input": ({"value": 0}, 5),    "output": 5},
        {"input": ({"value": 10}, 3),   "output": 13},
        # Adversarial: zero increments — naive agents may crash or return wrong value
        {"input": ({"value": 7}, 0),    "output": 7},
        {"input": ({"value": 0}, 100),  "output": 100},
    ],

    "structure_rule": "uses_lock",
}


# ==============================================================
# HARD TASK 3 — Memory leak (unbounded list → generator)
# ==============================================================
HARD_TASK_3 = {
    "id": "hard_3",
    "bug_type": "performance",

    "code": """def process_log_lines(filepath, keyword):
    results = []
    with open(filepath) as f:
        for line in f:
            if keyword in line:
                results.append(line.strip())
    return results""",

    "description": (
        "This function reads a log file and collects matching lines into a list. "
        "For very large log files (GBs) this causes a memory leak — the entire "
        "result set is held in RAM. Refactor it into a generator function using "
        "'yield' so lines are produced lazily one at a time. "
        "The function name and signature must stay the same."
    ),

    "expected_solution": """def process_log_lines(filepath, keyword):
    with open(filepath) as f:
        for line in f:
            if keyword in line:
                yield line.strip()""",

    "hint": "Replace 'results = []' and 'results.append(...)' with 'yield line.strip()'. This makes it a generator that yields one matching line at a time instead of loading all into memory.",

    # Test cases use a pre-written temp file via a helper trick:
    # inputs are (filepath, keyword) but safe_execute can't open files,
    # so we use structure_rule to grade this task structurally.
    # Correctness is verified by checking "uses_yield" in AST.
    "test_cases": [
        # These are placeholder correctness checks — primary grading is structural
        # A real FastAPI deployment would write a temp file and test the generator.
        # For now: agent must produce valid Python that compiles and uses yield.
    ],

    "structure_rule": "uses_yield",
}