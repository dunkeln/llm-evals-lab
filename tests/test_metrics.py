from src.core.metrics import *

def test_token_f1_metrics():
    assert token_f1("a b c", "a b c") == 1.0
    assert token_f1("a b", "a b c") > 0.0
    assert token_f1("", "a b c") == 0.0
