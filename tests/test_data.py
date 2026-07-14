from src.data.akshare_loader import _fmt


def test_fmt_scales_to_yi():
    # 大额数字转「亿」，便于研报阅读
    assert _fmt(1720.54e8).endswith("亿")
    assert "亿" in _fmt(823_200_000_00)


def test_fmt_small_number_keeps_plain():
    assert _fmt(24.5) == "24.50"


def test_fmt_non_numeric_passthrough():
    assert _fmt("N/A") == "N/A"
