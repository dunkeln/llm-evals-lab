from src.core.metrics import *
from unittest.mock import patch

def test_token_f1_metrics():
    assert token_f1("a b c", "a b c") == 1.0
    assert token_f1("a b", "a b c") > 0.0
    assert token_f1("", "a b c") == 0.0


def test_token_f1_perfect_match():
    assert token_f1("The cat sat", "the cat sat") == 1.0

def test_token_f1_partial_overlap():
    score = token_f1("the red cat", "the blue cat")
    assert 0 < score < 1

def test_token_f1_no_overlap():
    assert token_f1("cat", "dog") == 0.0

def test_cosine_similarity_range():
    sim = cosine_similarity("cat", "cat")
    assert 0.99 <= sim <= 1.0  # or just 0 <= sim <= 1

def test_cosine_similarity_different_texts():
    a = cosine_similarity("cat", "dog")
    b = cosine_similarity("cat", "quantum mechanics")
    assert 0 <= a <= 1
    assert 0 <= b <= 1

def test_judgeLM_shape_and_ranges():
    fake_result = LLMJudgeResult(
        toxicity=1, verbosity=5, hallucination=2, correctness=1
    )

    with patch("src.core.metrics.judgeLM") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.with_structured_output.return_value.invoke.return_value = fake_result

        res = judgeLM("2 + 2 = 4", "The answer is 4")

    assert res.toxicity != 1
    assert 1 <= res.verbosity <= 10
    assert 0 <= res.hallucination <= 10
    assert res.correctness in (0, 1)
