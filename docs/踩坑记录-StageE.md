# Stage E 踩坑记录（2026-07-15，本地微调 bge-reranker 全程实录）

> 环境：MacBook Air M2 / 16GB / macOS，热点+VPN 弱网。
> 每一条都是当天真实踩到、定位并修复的问题。格式：现象 → 根因 → 解法 → 教训。
> 面试时可作为「工程实战经验」素材，比"跑通了 demo"有说服力得多。

---

## 一、Python 环境类

### 1. conda base 遮蔽项目 venv
- **现象**：终端明明显示 `(.venv)`，`python main.py` 却报 `ModuleNotFoundError: No module named 'langgraph'`。
- **根因**：提示符显示 `(.venv) (base)` 双前缀时，conda 的 `/opt/anaconda3/bin` 在 PATH 里排在 venv 前面，`python` 实际解析到 conda 的解释器。
- **解法**：`conda deactivate` 后重新 `source .venv/bin/activate`；或永远用绝对路径 `.venv/bin/python`。根治：`conda config --set auto_activate_base false`。
- **教训**：不确定时先 `python -c "import sys; print(sys.executable)"` 验明正身。

### 2. venv 内 CLI 工具 shebang 失效
- **现象**：`.venv/bin/hf`、`.venv/bin/pip` 报 `bad interpreter: .../旧目录名/.venv/bin/python3.11: no such file or directory`。
- **根因**：venv 创建后项目目录改过名，console-script 的 shebang 写死了旧绝对路径。
- **解法**：一律改用 `python -m` 形式（`.venv/bin/python -m pip`、`python -m huggingface_hub ...`），不依赖 shebang。
- **教训**：venv 不可移动/改名；`python -m` 永远比入口脚本可靠。

---

## 二、模型下载类（弱网 + VPN 环境）

### 3. HF 新版 Xet 传输协议与镜像不兼容
- **现象**：`snapshot_download` 走 hf-mirror 卡在 0%，几分钟 0 字节。
- **根因**：新版 huggingface_hub 默认走 Xet CAS 协议，镜像站不支持，连接建立后无数据。
- **解法**：`HF_HUB_DISABLE_XET=1`；或干脆 curl 直连 `resolve/main/<file>` 文件 URL。

### 4. CDN 忽略 Range 请求 → 断点续传变"从头覆盖"
- **现象**：下载进度不升反降（88MB → 74MB），最终文件尺寸对但内容损坏（加载段错误）。
- **根因**：`curl -C -` 重连时部分 CDN 节点返回 200 全量而非 206 分段，curl 截断重写；错误页（HTML）也可能被当内容写入。
- **解法**：自写分块下载器（[scripts/robust_download.py](../scripts/robust_download.py)）：8MB 一块发 Range 请求、**强制校验 HTTP 206**（不是 206 直接作废换源）、进度单调不回退；完成后 **SHA256 对照官方哈希**（HF 的 LFS pointer / ModelScope API 都提供）。
- **教训**：弱网下大文件必须"分块 + 校验状态码 + 哈希验收"三件套；"文件大小对"≠"文件是好的"。

### 5. requests 的 timeout 防不住"滴字节"慢连接
- **现象**：下载进程活着但文件几分钟不涨，重试也不触发。
- **根因**：`requests` 的 timeout 是**字节间隔超时**——服务器每 59 秒滴 1 个字节连接就永不超时，一个 8MB 块能吊死几小时。
- **解法**：`stream=True` 流式读取 + **整块 wall-clock 死线**（90 秒到点砍连接）；砍掉时已收到的半块字节是有效前缀，保留写盘不浪费。
- **教训**：网络代码要区分"连接超时 / 读超时 / 总时长限制"，三者缺一不可。

### 6. 双镜像源轮换的前提是"字节级相同"
- **风险**：ModelScope 和 hf-mirror 轮换续传同一文件，若两站文件版本不同，拼出来的是杂交废件。
- **解法**：先比对两站的 SHA256（ModelScope 有 `/api/v1/models/<id>/repo/files` 接口），一致才允许混用。

### 7. 单源限流 → 换 CDN 比优化重连有效
- **现象**：hf-mirror 从 1MB/s 掉到 13KB/s，且新连接同样慢（被按 IP 限流）。
- **解法**：切换到 ModelScope（阿里独立 CDN），速度立刻恢复。多准备一条下载路线。

### 8. VPN 让国内源绕道海外
- **现象**：连百度都要 4.6 秒；hf-mirror / ModelScope 全部龟速 + 频繁 SSLError。
- **根因**：全局 VPN 把国内流量也送出国再绕回来，隧道抖动即 SSL 重置。
- **解法**：下载国内源时关 VPN 或用规则分流（绕过大陆）模式。

### 9. macOS 系统 Python 的 urllib 没有 SSL 证书
- **现象**：`urllib.request` 一律 `SSL: CERTIFICATE_VERIFY_FAILED`，curl 却正常。
- **根因**：python.org 安装的 Python 未执行 `Install Certificates.command`，urllib 无 CA 可用；curl 用系统钥匙串。
- **解法**：用 `requests`（自带 certifi）替代 urllib。

---

## 三、深度学习栈兼容性类（macOS Apple Silicon）

### 10. transformers 5.x 新版权重加载器在 macOS 段错误
- **现象**：`AutoModel.from_pretrained` 直接 Segmentation fault（exit 139），无 Python 异常。
- **根因**：transformers 5.13 的多线程权重物化（`core_model_loading._materialize_copy`）在 macOS + torch 2.13 组合下崩溃。
- **解法**：降级 `transformers==4.51.3`。
- **教训**：exit code 139 = 段错误 = C 层问题，Python try/except 接不住；用 `PYTHONFAULTHANDLER=1` + `python -u` 拿崩溃堆栈。

### 11. 新版 FlagEmbedding 只兼容 transformers 5.x API
- **现象**：降级 transformers 后 `BertModel.__init__() got an unexpected keyword argument 'dtype'`。
- **根因**：FlagEmbedding 新版按 transformers 5 的 `dtype=` 参数调用，4.x 只认 `torch_dtype=`。
- **解法**：配套降级 `FlagEmbedding==1.2.11`（与 transformers 4.x 的经典稳定组合）。
- **教训**：深度学习依赖要"成组降级"，锁版本组合而不是单个包。

### 12. faiss 与 torch 双 OpenMP 运行时冲突
- **现象**：单独用 FlagModel 正常、单独用 faiss 正常，两者同进程（建索引）即段错误。
- **根因**：faiss-cpu 和 torch 各自静态链接了一份 libomp，同进程加载两份 OpenMP 运行时是未定义行为。
- **解法**：`import faiss` 之前设置 `KMP_DUPLICATE_LIB_OK=TRUE` + `OMP_NUM_THREADS=1`（已固化进 [src/rag/index.py](../src/rag/index.py)）。
- **教训**：macOS 上 faiss+torch 是著名组合坑，两个环境变量缺一不可。

---

## 四、数据与外部接口类

### 13. akshare 接口无超时，会永久挂起
- **现象**：批量拉 20 支股票，卡在第 10 支不动，整个 build 停摆。
- **根因**：akshare 底层 requests 不设超时，接口抽风时无限等待。
- **解法**：每支股票放独立子进程跑，`subprocess.run(..., timeout=90)` 硬超时，拉不动就跳过（17/20 成功即够用）。
- **教训**：批处理任务的单元之间要故障隔离；外部接口永远假设它会挂起。

### 14. chunk 文本缺公司名 → 检索"指鹿为马"（当天最重要的一课）
- **现象**：真实语义检索上线后，问"贵州茅台的ROE"命中长江电力、中国中免；问"宁德时代"前三名全是别家公司。
- **根因**：chunk 文本以裸代码开头（`600519 截至...`），而用户用公司名提问。通用 embedding 模型不可能知道"贵州茅台=600519"。
- **解法**：加代码→名称映射，chunk 文本改为 `贵州茅台(600519) 截至...`。重建索引后三类提问全部命中正确公司，相似度 0.5→0.8。
- **教训**：**检索效果差先查喂给模型的数据，再怀疑模型**。一行数据修复的收益超过任何调参。

### 15. 评测集无区分度 → 三组配置分数完全相同
- **现象**：baseline / 通用重排 / 微调重排三行指标一模一样。
- **根因**：8 题全问茅台，而库里茅台恰好只有 5 个 chunk = k_final，召回阶段就"满配"，排序无从发挥。
- **解法**：评测集升级为三梯度 15 题：多公司单点 / 跨公司对比（top-5 须同时容纳两家）/ 无公司名筛选（纯语义排序）。升级后差异立现。
- **教训**：**评测集必须制造区分压力**，否则测不出组件价值；"指标相同"本身也是有效信号（说明瓶颈在别处）。

---

## 五、工程习惯类

### 16. 管道会吞掉进度与错误
- **现象**：后台任务"完成"但零输出；错误码被 `| tail` 吃掉显示 exit 0。
- **根因**：stdout 缓冲 + 管道退出码取最后一环；段错误时缓冲区内容直接丢失。
- **解法**：长任务用 `python -u`（不缓冲）+ 输出重定向到日志文件（不经管道）；判断成败看显式 `echo exit=$?`。

### 17. 下载/训练必须配独立监控
- **做法**：Monitor 盯文件大小增量（每 45-60 秒报一次，停滞报警）；训练盯 loss 行 + 崩溃关键字（Traceback/Killed/Segmentation）。
- **教训**："沉默"和"正常运行"在外部看起来一样，必须有心跳信号。

---

## 附：当天最终成果

- 微调 bge-reranker-base（278M，MPS，2 epoch ≈ 25 分钟）：dev acc@1 **0.879 → 1.000**
- 端到端评估（15 题三梯度）：通用重排在金融域**负迁移**（faithfulness 0.80→0.64），
  微调后收复反超，相对通用重排 context_precision / faithfulness 均 **+24%**
- 详见 [eval_results.md](../eval_results.md) 与 [README.md](../README.md)
