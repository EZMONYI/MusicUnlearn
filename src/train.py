import argparse
import logging
import os
import sys
from collections import OrderedDict

import torch
import yaml
from torch.utils.data import DataLoader

from routines.scrub import scrub
from utils.dict import Dictionary
from utils.helpers import build_datasets
from utils.model import (
    build_model,
)


class Config(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def main(args):
    # assert (
    #     args.max_tokens is not None or args.batch_size is not None
    # ), "Must specify batch size either with --max-tokens or --batch-size"
    dicts = OrderedDict()

    for lang in args.langs:
        dicts[lang] = Dictionary.load(
            os.path.join(args.data, "dict.{}.txt".format(lang))
        )
        print("| [{}] dictionary: {} types".format(lang, len(dicts[lang])))

    teacher = build_model(args, dicts)
    student = build_model(args, dicts)
    dataset = build_datasets(args, dicts, "train", args.training)
    dataset.ordered_indices()

    dataloader = DataLoader(
        dataset, shuffle=True, batch_size=8, collate_fn=dataset.collater
    )
    ckpt = torch.load(args.checkpoint_path)

    teacher.load_state_dict(ckpt, strict=False)
    student.load_state_dict(ckpt, strict=False)

    scrub(teacher, student, dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("-c", "--config")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    config = Config(config)
    main(config)
