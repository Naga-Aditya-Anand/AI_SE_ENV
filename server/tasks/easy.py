EASY_TASK = {
    "id": "easy_1",
    "bug_type": "syntax",

    "code": """def add(a, b)
    return a + b""",

    "description": "Fix the syntax error in this function.",

    "expected_solution": """def add(a, b):
    return a + b""",

    "hint": "Look at the function definition line — it's missing a colon (:) after the parameter list.",

    "test_cases": [
        {"input": (2, 3),  "output": 5},
        {"input": (5, 7),  "output": 12},
        # Adversarial: negative numbers — naive copy-paste of the bug won't handle these
        {"input": (-1, -1), "output": -2},
        {"input": (0, 0),   "output": 0},
    ]
}


EASY_TASK_2 = {
    "id": "easy_2",
    "bug_type": "type",

    "code": """def multiply(a, b):
    return str(a * b)""",

    "description": (
        "This function should return the product of two numbers as an integer, "
        "but it is returning a string instead. Fix the return type bug."
    ),

    "expected_solution": """def multiply(a, b):
    return a * b""",

    "hint": "The function is wrapping the result in str(). Remove the str() call to return the integer product directly.",

    "test_cases": [
        {"input": (3, 4),   "output": 12},
        {"input": (5, 0),   "output": 0},
        # Adversarial: '12' == 12 is False in Python — catches agents that leave str()
        {"input": (6, 7),   "output": 42},
        {"input": (-2, 3),  "output": -6},
    ]
}