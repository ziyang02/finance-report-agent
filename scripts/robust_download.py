"""分块断点下载器：专治不稳定网络 + CDN 断点续传不可靠。

    python scripts/robust_download.py <url> <输出路径> [url2 ...]

原理：按 8MB 分块发 Range 请求，逐块落盘、逐块重试；
强制要求 206（服务器忽略 Range 就换下一个源），进度单调不回退。
多个 url 参数互为镜像源，轮换使用。
"""
from __future__ import annotations

import sys
import time

import requests  # 自带 certifi 证书；系统 Python 的 urllib 常因证书未装而 SSL 失败

CHUNK = 8 * 1024 * 1024
MAX_RETRY_PER_CHUNK = 30
_UA = {"User-Agent": "Mozilla/5.0"}


def total_size(urls: list[str]) -> int:
    """用 1 字节 Range 请求探测文件总大小（从 Content-Range 里读）。"""
    for url in urls:
        try:
            r = requests.get(url, headers={**_UA, "Range": "bytes=0-0"}, timeout=15)
            cr = r.headers.get("Content-Range", "")  # e.g. bytes 0-0/95081731
            if "/" in cr:
                return int(cr.split("/")[-1])
        except Exception as e:  # noqa: BLE001
            print(f"  探测失败 {url[:60]}… ({type(e).__name__})")
    raise RuntimeError("所有源都无法探测文件大小")


def fetch_range(url: str, start: int, end: int, deadline: float = 90.0) -> bytes:
    """带整块死线的流式下载。requests 的 timeout 只管字节间隔，
    服务器慢速"滴字节"能把连接吊死几小时——所以必须限制整块总时长。"""
    r = requests.get(url, headers={**_UA, "Range": f"bytes={start}-{end}"},
                     timeout=(10, 30), stream=True)
    if r.status_code != 206:     # 服务器忽略 Range -> 这次作废，避免覆盖进度
        raise RuntimeError(f"服务器不支持断点(HTTP {r.status_code})")
    buf, t0 = b"", time.time()
    for part in r.iter_content(chunk_size=256 * 1024):
        buf += part
        if time.time() - t0 > deadline:
            break                # 超死线：半块也是有效前缀字节，保留进度不浪费
    if not buf:
        raise RuntimeError("整块超时且0字节")
    return buf


def download(urls: list[str], out: str) -> None:
    size = total_size(urls)
    from pathlib import Path
    path = Path(out)
    done = path.stat().st_size if path.exists() else 0
    if done >= size:
        print(f"已完整：{out} ({size/1e6:.1f}MB)")
        return
    print(f"目标 {size/1e6:.1f}MB，已有 {done/1e6:.1f}MB，续传中…")

    t0, base = time.time(), done
    with open(path, "ab") as f:
        while done < size:
            end = min(done + CHUNK, size) - 1
            for attempt in range(MAX_RETRY_PER_CHUNK):
                url = urls[attempt % len(urls)]      # 轮换镜像源
                try:
                    data = fetch_range(url, done, end)
                    f.write(data)
                    f.flush()
                    done += len(data)
                    speed = (done - base) / max(time.time() - t0, 1) / 1024
                    print(f"  {done/1e6:.1f}/{size/1e6:.1f}MB ({speed:.0f}KB/s)", flush=True)
                    break
                except Exception as e:  # noqa: BLE001
                    wait = min(2 * (attempt + 1), 20)
                    print(f"  块重试{attempt+1} ({type(e).__name__}) {wait}s后再试", flush=True)
                    time.sleep(wait)
            else:
                raise RuntimeError(f"块 {done}-{end} 重试{MAX_RETRY_PER_CHUNK}次仍失败")
    print(f"完成：{out}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    download([sys.argv[1]] + sys.argv[3:], sys.argv[2])
