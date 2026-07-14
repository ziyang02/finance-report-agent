from src.tools.finance_tools import calc_metrics, dispatch


def test_get_financials_mock():
    r = dispatch("get_financials", {"code": "600519"})
    assert r["name"] == "贵州茅台"
    assert r["roe"] > 0


def test_calc_metrics():
    r = calc_metrics(revenue=1000, net_profit=340, equity=1000)
    assert r["net_margin"] == 0.34
    assert r["roe"] == 0.34


def test_dispatch_unknown_tool():
    assert "error" in dispatch("no_such_tool", {})


def test_dispatch_accepts_json_string():
    # 用 get_financials（走本地 mock，不触网）验证 dispatch 能解析 JSON 字符串参数
    r = dispatch("get_financials", '{"code": "600519"}')
    assert r["code"] == "600519"
    assert r["name"] == "贵州茅台"
