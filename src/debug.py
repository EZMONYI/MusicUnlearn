from utils.helpers import load_data
from torch.utils.data import DataLoader
import argparse
import yaml
import os
from collections import OrderedDict
from utils.dict import Dictionary


class Config(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def main(args):
    dicts = OrderedDict()

    for lang in args.langs:
        dicts[lang] = Dictionary.load(
            os.path.join(args.data, "dict.{}.txt".format(lang))
        )
        print("| [{}] dictionary: {} types".format(lang, len(dicts[lang])))

    dataset = load_data(args, dicts, "train", args.training)
    dataset.ordered_indices()
    dataloader = DataLoader(
        dataset, batch_size=8, collate_fn=dataset.collater
    )
    for idx, sample in enumerate(dataloader):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("-c", "--config")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    config = Config(config)
    main(config)
