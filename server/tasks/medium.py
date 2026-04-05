MEDIUM_TASK = {
    "id": "medium_1",
    "bug_type": "logic",

    "code": """def average(arr):
    return sum(arr) / len(arr) - 1""",

    "description": "Fix the logical error in this function so it correctly computes the average.",

    "expected_solution": """def average(arr):
    return sum(arr) / len(arr)""",

    "hint": "The function is subtracting 1 from the result. The average formula is simply: sum / count. Remove the '- 1' part.",

    "test_cases": [
        {"input": ([1, 2, 3]),      "output": 2.0},
        {"input": ([10, 20, 30]),   "output": 20.0},
        # Adversarial: single element — catches agents who over-engineer and break simple cases
        {"input": ([5]),            "output": 5.0},
        {"input": ([0, 0, 0]),      "output": 0.0},
    ]
}


MEDIUM_TASK_2 = {
    "id": "medium_2",
    "bug_type": "off_by_one",

    "code": """def find_max(arr):
    max_val = arr[0]
    for i in range(1, len(arr) - 1):
        if arr[i] > max_val:
            max_val = arr[i]
    return max_val""",

    "description": (
        "This function is supposed to find the maximum value in a list, "
        "but it has an off-by-one error — it never checks the last element. Fix it."
    ),

    "expected_solution": """def find_max(arr):
    max_val = arr[0]
    for i in range(1, len(arr)):
        if arr[i] > max_val:
            max_val = arr[i]
    return max_val""",

    "hint": "The loop range is range(1, len(arr) - 1) which stops one element before the end. Change it to range(1, len(arr)) to include all elements.",

    "test_cases": [
        # The max is deliberately placed at the LAST index to expose the off-by-one bug
        {"input": ([1, 3, 2, 5]),       "output": 5},
        {"input": ([10, 20, 30, 99]),   "output": 99},
        # Adversarial: max is in the middle — catches agents who only fix the last element check
        {"input": ([1, 100, 2, 3]),     "output": 100},
        # Adversarial: single element list
        {"input": ([42]),               "output": 42},
    ]
}