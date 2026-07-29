"""実験2: HTTP Range Requestでwheel全体をダウンロードせずに
zipの central directory だけを見て *.dist-info/METADATA だけを取り出すデモ。

PEP658の.metadataサイドカーファイルを一切使わず、"生"のRangeリクエストのみで
どれだけの転送量でメタデータが取得できるかを実証する。
"""
import io
import json
import sys
import time
import urllib.request
import zipfile

WHEEL_URL = sys.argv[1] if len(sys.argv) > 1 else (
    "https://files.pythonhosted.org/packages/2f/69/7a1984df015d01875c9ac79bfda7c31492cc94f81a2625303ce4707188e5/"
    "streamlit-1.60.0-py3-none-any.whl"
)


class HTTPRangeFile:
    """zipfile.ZipFile が要求するファイルインタフェース(read/seek/tell)を
    HTTP Range Request で実装するラッパー。実際に転送したバイト数と
    リクエスト回数を記録する。
    """

    def __init__(self, url):
        self.url = url
        self.pos = 0
        self.total_transferred = 0
        self.request_count = 0
        self.size = self._head_size()

    def _head_size(self):
        req = urllib.request.Request(self.url, method="HEAD")
        with urllib.request.urlopen(req) as resp:
            return int(resp.headers["Content-Length"])

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        start = self.pos
        end = min(self.pos + n, self.size) - 1
        if end < start:
            return b""
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        dt = time.perf_counter() - t0
        self.total_transferred += len(data)
        self.request_count += 1
        self.pos += len(data)
        print(
            f"  [range-request #{self.request_count}] bytes={start}-{end} "
            f"({len(data)} bytes, {dt:.3f}s)",
            file=sys.stderr,
        )
        return data

    def readable(self):
        return True

    def seekable(self):
        return True


def main():
    print(f"target wheel: {WHEEL_URL}", file=sys.stderr)
    rf = HTTPRangeFile(WHEEL_URL)
    wheel_total_size = rf.size
    print(f"wheel total size (HEAD): {wheel_total_size} bytes", file=sys.stderr)

    with zipfile.ZipFile(rf) as zf:
        names = zf.namelist()
        meta_name = next(n for n in names if n.endswith(".dist-info/METADATA"))
        print(f"found metadata entry: {meta_name}", file=sys.stderr)
        metadata_bytes = zf.read(meta_name)

    result = {
        "wheel_url": WHEEL_URL,
        "wheel_total_size_bytes": wheel_total_size,
        "metadata_entry_name": meta_name,
        "metadata_content_size_bytes": len(metadata_bytes),
        "total_transferred_bytes_via_range": rf.total_transferred,
        "request_count": rf.request_count,
        "transferred_over_wheel_ratio": round(rf.total_transferred / wheel_total_size, 6),
        "metadata_first_10_lines": metadata_bytes.decode("utf-8", errors="replace").splitlines()[:10],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
