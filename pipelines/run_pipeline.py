"""Entry point để chạy pipeline hai stage với config đơn giản.

Đây là demo tối thiểu dùng dữ liệu giả (toy data). Sau này bạn có thể:
- Thay thế phần load train_data bằng dữ liệu thật (từ dataset.pkl)
- Thêm logging, metrics, evaluation, ...
"""

import argparse
from pathlib import Path

from pipelines.base import PipelineConfig, RetrievalConfig, RerankConfig, TwoStagePipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run two-stage recommendation pipeline")
    parser.add_argument("--retrieval_method", type=str, default="lrurec")
    parser.add_argument("--retrieval_top_k", type=int, default=200)
    parser.add_argument(
        "--rerank_method",
        type=str,
        default="identity",
        help="Tên reranker (vd: identity, random, vip4, ...). Dùng 'none' để chỉ chạy Stage 1.",
    )
    parser.add_argument("--rerank_top_k", type=int, default=50)
    return parser.parse_args()


def build_pipeline_from_args(args: argparse.Namespace) -> TwoStagePipeline:
    retrieval_cfg = RetrievalConfig(method=args.retrieval_method, top_k=args.retrieval_top_k)
    rerank_cfg = RerankConfig(method=args.rerank_method, top_k=args.rerank_top_k)
    cfg = PipelineConfig(retrieval=retrieval_cfg, rerank=rerank_cfg)
    return TwoStagePipeline(cfg)


def main() -> None:
    args = parse_args()

    # TODO: thay bằng dữ liệu thật từ dataset.pkl
    # Toy train data: mỗi user có list items đã tương tác
    train_data = {
        1: [10, 11, 12, 13, 14],
        2: [20, 21, 22, 23, 24],
        3: [30, 31, 32, 33, 34],
    }

    pipeline = build_pipeline_from_args(args)
    pipeline.fit(train_data)

    for user_id in train_data.keys():
        recs = pipeline.recommend(user_id)
        print(f"User {user_id} → recommendations: {recs}")


if __name__ == "__main__":
    main()
