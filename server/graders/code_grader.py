import ast
import threading as _threading_module
from concurrent.futures import ThreadPoolExecutor, TimeoutError


def safe_execute(code_str, func_name, inputs, timeout_seconds=0.1, allow_threading=False):
    """
    Safely executes the submitted code in a sandboxed environment with a strict timeout.
    Returns (success: bool, result_or_error_message).

    Args:
        allow_threading: inject the threading module so agents can use
                         threading.Lock. Set True for structure_rule='uses_lock' tasks.
    """
    def target_execution():
        safe_builtins = {
            'list': list, 'set': set, 'dict': dict, 'tuple': tuple,
            'int': int, 'float': float, 'str': str, 'bool': bool,
            'len': len, 'range': range, 'sum': sum, 'min': min, 'max': max,
            'abs': abs, 'round': round, 'enumerate': enumerate, 'zip': zip,
            'map': map, 'filter': filter, 'sorted': sorted, 'isinstance': isinstance,
            'True': True, 'False': False, 'None': None,
        }
        if allow_threading:
            safe_builtins['__import__'] = __import__

        # Single namespace so module-level vars (e.g. _lock) are visible
        # inside function bodies defined in the same exec block
        namespace = {'__builtins__': safe_builtins}
        if allow_threading:
            namespace['threading'] = _threading_module

        exec(code_str, namespace)
        return namespace[func_name](*inputs)

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(target_execution)
            result = future.result(timeout=timeout_seconds)
            return True, result
    except TimeoutError:
        return False, "TIMEOUT: Code is not optimised enough for large inputs."
    except Exception as e:
        return False, str(e)


def extract_function_name(code_str):
    """
    Extracts the first top-level function name from submitted code using AST.
    Returns None if the code cannot be parsed or no function is found.
    """
    try:
        tree = ast.parse(code_str)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return node.name
        return None
    except Exception:
        return None


# ==============================================================
# Structural rule checkers
# ==============================================================

def has_nested_loops(code_str):
    """
    Returns True if the code contains a loop nested inside another loop.
    Used by structure_rule: 'no_nested_loops'.
    """
    try:
        tree = ast.parse(code_str)
        loop_nodes = (ast.For, ast.While, ast.AsyncFor)
        for node in ast.walk(tree):
            if isinstance(node, loop_nodes):
                for child in ast.walk(node):
                    if child is not node and isinstance(child, loop_nodes):
                        return True
        return False
    except Exception:
        return True  # Unparseable code fails the check


def uses_lock(code_str):
    """
    Returns True if the code uses threading.Lock (or RLock) via AST analysis.
    Catches: threading.Lock(), threading.RLock(), Lock() after 'from threading import Lock'.
    Used by structure_rule: 'uses_lock'.
    """
    try:
        tree = ast.parse(code_str)
        for node in ast.walk(tree):
            # Catches: threading.Lock() / threading.RLock()
            if isinstance(node, ast.Attribute):
                if node.attr in ("Lock", "RLock"):
                    return True
            # Catches: Lock() after 'from threading import Lock'
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in ("Lock", "RLock"):
                    return True
            # Catches: 'with lock:' or 'with threading.Lock() as ...'
            if isinstance(node, ast.With):
                return True
        return False
    except Exception:
        return False


def uses_yield(code_str):
    """
    Returns True if the code contains a 'yield' statement, making it a generator.
    Used by structure_rule: 'uses_yield'.
    """
    try:
        tree = ast.parse(code_str)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Yield, ast.YieldFrom)):
                return True
        return False
    except Exception:
        return False


def measure_code_quality(code_str):
    """
    Returns a quality score in [0.0, 1.0] based on AST metrics.
    Used by the 'refactor' action type.

    Metrics:
      - Function length (lines): penalise if > 20 lines
      - Variable count: penalise if > 10 local variables
      - Nesting depth: penalise deep if/for nesting
    """
    try:
        tree = ast.parse(code_str)
        lines = code_str.strip().split("\n")
        line_score = max(0.01, min(1.0 - max(0, len(lines) - 20) * 0.05, 0.99))

        # Count assigned variable names
        assigned_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assigned_names.add(target.id)
        var_score = max(0.01, min(1.0 - max(0, len(assigned_names) - 10) * 0.1, 0.99))
        quality = (line_score + var_score) / 2
        return round(max(0.01, min(quality, 0.99)), 3)
    except Exception:
        return 0.5  # Neutral fallback


# ==============================================================
# Main grader
# ==============================================================

def grade_code(task, agent_output, action_type="fix"):
    """
    Grades submitted code and returns (score: float, feedback: str).

    Scoring pillars:
      Syntax check   → 0.2
      Test cases     → 0.6  (or 0.5 when structure_rule is active)
      Structure rule → 0.2  (only when task defines structure_rule)
      Quality bonus  → up to +0.1 extra when action_type == 'refactor'

    action_type:
      'fix'     → standard grading
      'refactor'→ standard grading + code quality bonus
      'review'  → handled by env.py before reaching grader (score always 0)
    """
    test_cases = task.get("test_cases", [])
    total = len(test_cases)
    structure_rule = task.get("structure_rule")
    score = 0.0
    feedback_msgs = []

    # ----------------------------------------------------------
    # 1. SYNTAX CHECK  (0.2)
    # ----------------------------------------------------------
    try:
        compile(agent_output, "<string>", "exec")
        score += 0.2
    except Exception as e:
        return 0.01, f"Syntax Error: {str(e)}"

    # ----------------------------------------------------------
    # 2. LOGIC CHECK — test cases  (0.8 or 0.6 with struct rule)
    # ----------------------------------------------------------
    # structure_rule tasks: 0.2 syntax + 0.6 tests + 0.2 struct = 1.0
    # no structure_rule:    0.2 syntax + 0.8 tests             = 1.0
    test_weight = 0.6 if structure_rule else 0.8

    if total > 0:
        passed = 0
        func_name = extract_function_name(agent_output)

        if not func_name:
            return 0.2, (
                "Error: Could not extract a function name. "
                "Make sure you are defining a standard Python function."
            )

        for i, test in enumerate(test_cases):
            inputs = (
                test["input"] if isinstance(test["input"], tuple) else (test["input"],)
            )
            allow_threading = (structure_rule == "uses_lock")
            success, result = safe_execute(agent_output, func_name, inputs, allow_threading=allow_threading)

            if success and result == test["output"]:
                passed += 1
            else:
                if not success:
                    feedback_msgs.append(
                        f"Test {i+1} Crashed  | Input: {inputs} | Error: {result}"
                    )
                else:
                    feedback_msgs.append(
                        f"Test {i+1} Failed   | Input: {inputs} "
                        f"| Expected: {test['output']} | Got: {result}"
                    )

        test_ratio = passed / total
        score += test_ratio * test_weight
    else:
        # No test cases — full test weight awarded (graded purely structurally)
        score += test_weight

    # ----------------------------------------------------------
    # 3. STRUCTURE RULE CHECK  (0.2, only when defined)
    # ----------------------------------------------------------
    if structure_rule == "no_nested_loops":
        if not has_nested_loops(agent_output):
            score += 0.2
        else:
            feedback_msgs.append(
                "Optimisation Failed: Your code contains nested loops (O(n²)). "
                "Please optimise to O(n)."
            )

    elif structure_rule == "uses_lock":
        if uses_lock(agent_output):
            score += 0.2
        else:
            feedback_msgs.append(
                "Thread-Safety Failed: Your code does not use threading.Lock. "
                "Wrap the counter increment in a 'with lock:' block."
            )

    elif structure_rule == "uses_yield":
        if uses_yield(agent_output):
            score += 0.2
        else:
            feedback_msgs.append(
                "Memory Efficiency Failed: Your code does not use 'yield'. "
                "Refactor the function into a generator to avoid loading all lines into RAM."
            )

    # ----------------------------------------------------------
    # 4. QUALITY BONUS for 'refactor' action  (up to +0.1)
    # ----------------------------------------------------------
    if action_type == "refactor":
        quality = measure_code_quality(agent_output)
        bonus = quality * 0.1
        score += bonus
        feedback_msgs.append(f"Code Quality Score: {quality:.2f} → bonus +{bonus:.2f}")

    # ----------------------------------------------------------
    # Final feedback assembly
    # ----------------------------------------------------------
    final_feedback = (
        "\n".join(feedback_msgs) if feedback_msgs else "Success! All tests passed."
    )

    # Ensure score is strictly between 0 and 1 (not 0.0 and not 1.0)
    final_score = round(score, 4)
    if final_score >= 1.0:
        final_score = 0.99
    elif final_score <= 0.0:
        final_score = 0.01
    return final_score, final_feedback