from datasets import *
from datasets.clip_embeddings import maybe_extract_clip_embeddings
from config import *
from pytorch_lightning import seed_everything


def main(args):
    seed_everything(args.seed)
    dataset = dataset_factory(args)
    # Cần gọi load_dataset() hoặc preprocess() để trigger việc download
    data = dataset.load_dataset()
    # Ngay sau khi dữ liệu (và ảnh) đã sẵn sàng, trích xuất CLIP embeddings
    # cho image/text nếu được yêu cầu.
    maybe_extract_clip_embeddings(dataset, data, args)

if __name__ == "__main__":
    main(arg)