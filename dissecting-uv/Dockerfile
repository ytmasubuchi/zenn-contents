# 「uvを解剖する」記事のベンチマーク用Dockerfile
#
# 使い方 (再現手順):
#   cd dissecting-uv
#   docker build -t uv-bench .
#   docker run -d --name uvbench -v $(pwd)/results:/bench/results uv-bench sleep infinity
#   docker exec uvbench python bench/exp1_install.py ...  (各実験スクリプトを実行)
#   docker rm -f uvbench
#
# ネットワークI/Oに依存する計測のため、コンテナは常駐させ、
# 各実験スクリプトを docker exec で個別に叩く運用とする。

FROM python:3.12-slim

WORKDIR /bench

# curl: PEP658 metadataやHTTP/2確認に使用。ca-certificates: TLS。
# jq: JSON整形。file: zip解析補助。procps: ps等(デバッグ用)。git: 一部ツールが要求する場合がある。
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    jq \
    file \
    procps \
    coreutils \
    && rm -rf /var/lib/apt/lists/*

# --- pip: python:3.12-slim に同梱されているものをそのまま使う(バージョンは実行時に記録) ---

# --- poetry ---
# 公式インストーラでシステム全体に導入する。--versionを指定せず、インストーラのデフォルト(最新安定版)を
# そのまま使う(バージョンは実行時に記録)。
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:${PATH}"

# --- uv ---
# 公式インストーラで導入する(バージョンは実行時に記録)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# キャッシュ/venv用の作業ディレクトリはコンテナ内(ホストは汚さない)
ENV PIP_CACHE_DIR=/bench/.cache/pip
ENV POETRY_CACHE_DIR=/bench/.cache/pypoetry
ENV UV_CACHE_DIR=/bench/.cache/uv
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

COPY bench/ ./bench/

CMD ["sleep", "infinity"]
