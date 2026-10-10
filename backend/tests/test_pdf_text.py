from src.core.pdf_builder import prettify_ascii_math


def test_keeps_brackets_and_words_intact():
    out = prettify_ascii_math("Common pitfalls: the expression my_list[-2] and {\"a\": 1}")
    assert "pitfalls" in out
    assert "expression" in out
    assert "my_list[-2]" in out
    assert '{"a": 1}' in out


def test_still_prettifies_math_words():
    assert prettify_ascii_math("area = pi r^2") == "area = π r²"
    assert prettify_ascii_math("sqrt(x) and theta") == "√(x) and θ"
