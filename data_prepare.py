from datasets import *
from config import *
from pytorch_lightning import seed_everything


def main(args):
    seed_everything(args.seed)
    dataset = dataset_factory(args)
    # Cần gọi load_dataset() hoặc preprocess() để trigger việc download
    dataset.load_dataset()

if __name__ == "__main__":
    main(arg)