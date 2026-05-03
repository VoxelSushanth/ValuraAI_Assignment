def normalize_ticker(ticker: str) -> str:
    """Normalize ticker: uppercase, strip exchange suffix"""
    return ticker.strip().upper().split(".")[0]


def amount_matches(expected: float, actual: float, tolerance: float = 0.05) -> bool:
    """Check if amounts match within tolerance"""
    if expected == 0:
        return abs(actual) < 0.001
    return abs(actual - expected) / abs(expected) <= tolerance


def normalize_string(s: str) -> str:
    """Normalize string for comparison"""
    return s.strip().lower()


def tickers_subset_match(expected: list[str], actual: list[str]) -> bool:
    """Check if all expected tickers are in actual (normalized)"""
    normalized_actual = {normalize_ticker(t) for t in actual}
    return all(normalize_ticker(e) in normalized_actual for e in expected)


def amounts_subset_match(expected: list[float], actual: list[float], tolerance: float = 0.05) -> bool:
    """Check if all expected amounts have a match in actual within tolerance"""
    return all(any(amount_matches(e, a, tolerance) for a in actual) for e in expected)


def string_list_subset_match(expected: list[str], actual: list[str]) -> bool:
    """Check if all expected strings are in actual (normalized)"""
    normalized_actual = {normalize_string(s) for s in actual}
    return all(normalize_string(e) in normalized_actual for e in expected)


def entities_match(expected: dict, actual: dict) -> tuple[bool, list[str]]:
    """
    Compare expected and actual entity dictionaries.
    Returns (success, list_of_failure_messages)
    """
    failures = []
    
    # Check tickers
    exp_t = expected.get("tickers", [])
    act_t = actual.get("tickers", [])
    if exp_t and not tickers_subset_match(exp_t, act_t):
        missing = [t for t in exp_t if normalize_ticker(t) not in {normalize_ticker(a) for a in act_t}]
        failures.append(f"Missing tickers: {missing}. Actual: {act_t}")
    
    # Check amounts
    exp_a = expected.get("amounts", [])
    act_a = actual.get("amounts", [])
    if exp_a and not amounts_subset_match(exp_a, act_a):
        failures.append(f"Amount mismatch. Expected: {exp_a}. Actual: {act_a}")
    
    # Check other fields
    for field in ["time_periods", "sectors", "topics"]:
        exp_f = expected.get(field, [])
        act_f = actual.get(field, [])
        if exp_f and not string_list_subset_match(exp_f, act_f):
            failures.append(f"Missing {field}: {exp_f}. Actual: {act_f}")
    
    return (len(failures) == 0, failures)
