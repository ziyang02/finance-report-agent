"""金融工具 = Function Calling 的被调函数。

每个工具：①一个真实实现（尽量走 akshare）②akshare 不可用/无网时回退 mock，
保证脚手架开箱即跑。③一份 OpenAI tools JSON schema 供模型理解。

阶段 A 的目标：让模型自己决定调用哪个工具、传什么参数，执行权在我们手里。
"""
from __future__ import annotations

import json

# ---- mock 数据：无网/未装 akshare 时的回退，仅演示用 ----
_MOCK = {
    "600519": {"name": "贵州茅台", "price": 1680.0, "pe": 24.5, "pb": 8.9,
               "roe": 0.34, "gross_margin": 0.915, "revenue_yoy": 0.16},
    "000858": {"name": "五粮液", "price": 138.0, "pe": 16.2, "pb": 3.8,
               "roe": 0.25, "gross_margin": 0.758, "revenue_yoy": 0.11},
}


def get_stock_price(code: str) -> dict:
    """查最新股价。"""
    try:
        import akshare as ak

        df = ak.stock_zh_a_spot_em()
        row = df[df["代码"] == code].iloc[0]
        return {"code": code, "name": row["名称"], "price": float(row["最新价"])}
    except Exception:
        m = _MOCK.get(code, {})
        return {"code": code, "name": m.get("name", "未知"), "price": m.get("price"),
                "_source": "mock"}


def get_financials(code: str) -> dict:
    """查关键财务指标（PE/PB/ROE/毛利率/营收增速）。真实实现可组合多个 akshare 接口。"""
    # TODO(阶段 B): 用 ak.stock_financial_abstract / stock_a_indicator_lg 拼装
    m = _MOCK.get(code)
    if not m:
        return {"code": code, "error": "无该代码数据（mock 仅含 600519/000858）"}
    return {"code": code, "name": m["name"], "pe": m["pe"], "pb": m["pb"],
            "roe": m["roe"], "gross_margin": m["gross_margin"],
            "revenue_yoy": m["revenue_yoy"], "_source": "mock"}


def calc_metrics(revenue: float, net_profit: float, equity: float) -> dict:
    """由原始财务数字算派生指标，演示"工具做确定性计算、别让 LLM 硬算"。"""
    return {
        "net_margin": round(net_profit / revenue, 4) if revenue else None,
        "roe": round(net_profit / equity, 4) if equity else None,
    }


# ---- 注册表：名字 -> 函数 ----
REGISTRY = {
    "get_stock_price": get_stock_price,
    "get_financials": get_financials,
    "calc_metrics": calc_metrics,
}

# ---- OpenAI tools schema：模型据此决定调用 ----
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "查询某支 A 股的最新价格",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "6位股票代码，如 600519"}},
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_financials",
            "description": "查询某支 A 股的关键财务指标：PE/PB/ROE/毛利率/营收同比增速",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "6位股票代码"}},
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calc_metrics",
            "description": "根据营收、净利润、股东权益计算净利率和 ROE",
            "parameters": {
                "type": "object",
                "properties": {
                    "revenue": {"type": "number"},
                    "net_profit": {"type": "number"},
                    "equity": {"type": "number"},
                },
                "required": ["revenue", "net_profit", "equity"],
            },
        },
    },
]


def dispatch(name: str, arguments: str | dict) -> dict:
    """按模型给的名字+参数执行工具。arguments 可能是 JSON 字符串。"""
    args = json.loads(arguments) if isinstance(arguments, str) else arguments
    fn = REGISTRY.get(name)
    if not fn:
        return {"error": f"未知工具 {name}"}
    return fn(**args)
