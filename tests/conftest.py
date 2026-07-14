"""测试隔离：全程离线、确定性、不花钱、不触网。

必须在任何 src.* 导入前设置——conftest 最先加载，正好满足。
- RAG_FAKE_EMBED=1：embedder/reranker 用伪向量，不下载 ~2GB 模型
- 清空 LLM_API_KEY：LLM 走离线 stub，不调用付费 API（load_dotenv 默认不覆盖已存在的env）
"""
import os

os.environ["RAG_FAKE_EMBED"] = "1"
os.environ["LLM_API_KEY"] = ""  # 即使存在 .env 也强制走 stub，测试不烧额度
