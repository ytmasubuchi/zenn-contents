# このブログ記事のベンチマークを再現するためのDockerfile
# 使い方:
#   docker build -t pandas-bench .
#   docker run --rm -v $(pwd)/results:/bench/results -v $(pwd)/images:/bench/images pandas-bench
#
# 実行後、./results に *.csv / *.json、./images に *.png が出力されます。
FROM python:3.11-slim

WORKDIR /bench

RUN apt-get update && apt-get install -y --no-install-recommends fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bench/ ./bench/

CMD ["sh", "-c", "python bench/write_env_info.py && python bench/run_exp_a.py && python bench/run_exp_b1.py && python bench/run_exp_b2.py && python bench/make_plots.py"]
