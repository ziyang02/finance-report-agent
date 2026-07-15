"""用 akshare 拉 A 股财务数据，转成 RAG 可用的自然语言文本块。

设计原则：
- 每个数据源单独 try/except，一个接口挂了不拖垮整体（akshare 接口随版本/网络会变）。
- 结果本地缓存成 json，别每次跑都请求（会限流、会变）。
- 输出统一为 [{text, source}]，text 是给检索/LLM 看的自然语言，source 标注来源。
"""
from __future__ import annotations

import json
from pathlib import Path

from src.config import settings

# financial_analysis_indicator 里我们关心的比率列（按子串匹配，容忍列名微调）
_RATIO_COLS = [
    "销售毛利率", "销售净利率", "净资产收益率", "加权净资产收益率",
    "主营业务收入增长率", "净利润增长率", "资产负债率", "流动比率", "速动比率",
]
# financial_abstract 里关心的关键科目（按子串匹配指标名）
_ABSTRACT_KEYS = [
    "营业总收入", "营业收入", "归母净利润", "净利润", "扣非净利润",
    "基本每股收益", "每股净资产", "经营现金流",
]


# 代码->名称映射：chunk 文本里带上公司名，检索才能匹配"贵州茅台"这类自然语言提问
_CODE_NAMES = {
    "600519": "贵州茅台", "000858": "五粮液", "601318": "中国平安", "600036": "招商银行",
    "601398": "工商银行", "600900": "长江电力", "601899": "紫金矿业", "600309": "万华化学",
    "002415": "海康威视", "000333": "美的集团", "600276": "恒瑞医药", "300750": "宁德时代",
    "002594": "比亚迪", "601888": "中国中免", "600030": "中信证券", "000651": "格力电器",
    "601012": "隆基绿能", "600887": "伊利股份", "002714": "牧原股份", "601668": "中国建筑",
}


def _label(code: str) -> str:
    name = _CODE_NAMES.get(code, "")
    return f"{name}({code})" if name else code


def _cache_path(code: str) -> Path:
    d = Path(settings()["data"]["cache_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{code}.json"


def _fmt(v) -> str:
    try:
        f = float(v)
        return f"{f:,.2f}" if abs(f) < 1e8 else f"{f/1e8:.2f}亿"
    except (TypeError, ValueError):
        return str(v)


def _docs_from_analysis(code: str) -> list[dict]:
    """财务分析指标 -> 最近两期的关键比率文本。"""
    import akshare as ak

    df = ak.stock_financial_analysis_indicator(symbol=code, start_year="2023")
    df = df.sort_values("日期", ascending=False).head(2)
    docs = []
    for _, row in df.iterrows():
        date = row.get("日期", "?")
        parts = []
        for key in _RATIO_COLS:
            col = next((c for c in df.columns if key in c), None)
            if col and str(row[col]) not in ("nan", "None", ""):
                parts.append(f"{col}={row[col]}")
        if parts:
            docs.append({
                "text": f"{_label(code)} 截至 {date} 的关键财务比率：" + "；".join(parts) + "。",
                "source": f"{code}-财务比率-{date}",
            })
    return docs


def _docs_from_abstract(code: str) -> list[dict]:
    """财务摘要 -> 最近报告期的关键科目文本。"""
    import akshare as ak

    df = ak.stock_financial_abstract(symbol=code)
    date_cols = [c for c in df.columns if str(c).isdigit() and len(str(c)) == 8]
    date_cols = sorted(date_cols, reverse=True)[:2]
    docs = []
    for date in date_cols:
        parts = []
        for _, row in df.iterrows():
            name = str(row["指标"])
            if any(k in name for k in _ABSTRACT_KEYS):
                val = row.get(date)
                if str(val) not in ("nan", "None", ""):
                    parts.append(f"{name}={_fmt(val)}")
        if parts:
            docs.append({
                "text": f"{_label(code)} 报告期 {date} 的主要财务数据：" + "；".join(parts[:12]) + "。",
                "source": f"{code}-财务摘要-{date}",
            })
    return docs


def _docs_from_valuation(code: str) -> list[dict]:
    """个股估值 -> 最新一日 PE/PB/PEG/市值 文本（研报"估值"章节的数据来源）。"""
    import akshare as ak

    df = ak.stock_value_em(symbol=code)
    row = df.sort_values("数据日期").iloc[-1]
    date = row["数据日期"]
    parts = []
    for col in ("当日收盘价", "总市值", "PE(TTM)", "PE(静)", "市净率", "PEG值", "市销率"):
        if col in df.columns and str(row[col]) not in ("nan", "None", ""):
            val = _fmt(row[col]) if col in ("总市值", "当日收盘价") else round(float(row[col]), 2)
            parts.append(f"{col}={val}")
    if not parts:
        return []
    return [{
        "text": f"{_label(code)} 截至 {date} 的估值指标：" + "；".join(parts) + "。",
        "source": f"{code}-估值-{date}",
    }]


def load_company_docs(code: str, use_cache: bool = True) -> list[dict]:
    """返回 [{text, source}]，供 build_index 切块建库。"""
    cache = _cache_path(code)
    if use_cache and cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    docs: list[dict] = []
    sources = (("analysis", _docs_from_analysis), ("abstract", _docs_from_abstract),
               ("valuation", _docs_from_valuation))
    for name, fn in sources:
        try:
            got = fn(code)
            docs += got
            print(f"  [{code}] {name}: {len(got)} 段")
        except Exception as e:  # noqa: BLE001
            print(f"  [{code}] {name} 拉取失败({type(e).__name__})，跳过")

    if not docs:  # 全挂时回退占位，保证 build_index 不空跑
        docs = [{"text": f"公司代码{code}：数据拉取失败，占位。", "source": f"{code}-占位"}]

    cache.write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")
    return docs
