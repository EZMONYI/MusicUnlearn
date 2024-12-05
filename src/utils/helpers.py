import copy
import os
import re
from collections import Counter, OrderedDict

import numpy as np

from .mmap_dataset import DatasetBuilder, MMapDataset
from .music_mass_dataset import MusicMassDataset
from .music_mt_dataset import MusicMtDataset
from .round_robin_zip_dataset import RoundRobinZipDatasets

dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float32,
    7: np.double,
    8: np.uint16,
}


def read_longs(f, n):
    a = np.empty(n, dtype=np.int64)
    f.readinto(a)
    return a


def find_offsets(filename, num_chunks):
    with open(filename, "r", encoding="utf-8") as f:
        size = os.fstat(f.fileno()).st_size
        chunk_size = size // num_chunks
        offsets = [0 for _ in range(num_chunks + 1)]
        for i in range(1, num_chunks):
            f.seek(chunk_size * i)
            safe_readline(f)
            offsets[i] = f.tell()
        return offsets


def file_name(prefix, lang):
    fname = prefix
    if lang is not None:
        fname += ".{lang}".format(lang=lang)
    return fname


def index_file_path(prefix_path):
    return prefix_path + ".idx"


def data_file_path(prefix_path):
    return prefix_path + ".bin"


def best_fitting_dtype(vocab_size=None):
    if vocab_size is not None and vocab_size < 65500:
        return np.uint16
    else:
        return np.int32


def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins


def dataset_dest_prefix(args, output_prefix, lang):
    base = "{}/{}".format(args.destdir, output_prefix)
    if lang is not None:
        lang_part = ".{}-{}.{}".format(args.source_lang, args.target_lang, lang)
    elif args.only_source:
        lang_part = ""
    else:
        lang_part = ".{}-{}".format(args.source_lang, args.target_lang)

    return "{}{}".format(base, lang_part)


def dataset_dest_file(args, output_prefix, lang, extension):
    base = dataset_dest_prefix(args, output_prefix, lang)
    return "{}.{}".format(base, extension)


def tokenize_line(line):
    space_normalizer = re.compile(r"\s+")
    line = space_normalizer.sub(" ", line)
    line = line.strip()
    return line.split()


def binarize(
    filename,
    dict,
    consumer,
    tokenize=tokenize_line,
    append_eos=True,
    reverse_order=False,
    offset=0,
    end=-1,
):
    nseq, ntok = 0, 0
    replaced = Counter()

    def replaced_consumer(word, idx):
        if idx == dict.unk_index and word != dict.unk_word:
            replaced.update([word])

    with open(filename, "r", encoding="utf-8") as f:
        f.seek(offset)
        line = safe_readline(f)
        while line:
            if end > 0 and f.tell() > end:
                break
            else:
                ids = dict.encode_line(
                    line=line,
                    line_tokenizer=tokenize,
                    add_if_not_exist=False,
                    consumer=replaced_consumer,
                    append_eos=append_eos,
                    reverse_order=reverse_order,
                )
            nseq += 1
            ntok += len(ids)
            consumer(ids)
            line = f.readline()
    return {
        "nseq": nseq,
        "nunk": sum(replaced.values()),
        "ntok": ntok,
        "replaced": replaced,
    }


def build_ds(args, filename, vocab, output_prefix, lang, offset, end, append_eos=True):
    ds = DatasetBuilder(
        dataset_dest_file(args, output_prefix, lang, "bin"),
        dtype=best_fitting_dtype(len(vocab)),
    )

    def consumer(tensor):
        ds.add_item(tensor)

    res = binarize(
        filename,
        vocab,
        lambda t: ds.add_item(t),
        append_eos=append_eos,
        offset=offset,
        end=end,
    )
    ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))
    return res


def collate_tokens(
    values,
    pad_idx,
    eos_idx=None,
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res


def _match_types(arg1, arg2):
    """Convert the numerical argument to the same type as the other argument"""

    def upgrade(arg_number, arg_structure):
        if isinstance(arg_structure, tuple):
            return tuple([arg_number] * len(arg_structure))
        elif isinstance(arg_structure, dict):
            arg = copy.deepcopy(arg_structure)
            for k in arg:
                arg[k] = upgrade(arg_number, arg_structure[k])
            return arg
        else:
            return arg_number

    if isinstance(arg1, float) or isinstance(arg1, int):
        return upgrade(arg1, arg2), arg2
    elif isinstance(arg2, float) or isinstance(arg2, int):
        return arg1, upgrade(arg2, arg1)

    return arg1, arg2


def resolve_max_positions(*args):
    """Resolve max position constraints from multiple sources."""

    def map_value_update(d1, d2):
        updated_value = copy.deepcopy(d1)
        for key in d2:
            if key not in updated_value:
                updated_value[key] = d2[key]
            else:
                updated_value[key] = min(d1[key], d2[key])
        return updated_value

    def nullsafe_min(l):
        minim = None
        for item in l:
            if minim is None:
                minim = item
            elif item is not None and item < minim:
                minim = item
        return minim

    max_positions = None
    for arg in args:
        if max_positions is None:
            max_positions = arg
        elif arg is not None:
            max_positions, arg = _match_types(max_positions, arg)
            if isinstance(arg, float) or isinstance(arg, int):
                max_positions = min(max_positions, arg)
            elif isinstance(arg, dict):
                max_positions = map_value_update(max_positions, arg)
            else:
                max_positions = tuple(map(nullsafe_min, zip(max_positions, arg)))

    return max_positions


def _get_mass_dataset_key(lang_pair):
    return "mass:" + lang_pair


def _get_mt_dataset_key(lang_pair):
    return "" + lang_pair


def load_data(args, dicts, split, training):
    def split_exists(split, lang):
        filename = os.path.join(args.data, "{}.{}".format(split, lang))
        return os.path.exists(index_file_path(filename)) and os.path.exists(
            data_file_path(filename)
        )

    def split_para_exists(split, key, lang):
        filename = os.path.join(args.data, "{}.{}.{}".format(split, key, lang))
        return os.path.exists(index_file_path(filename)) and os.path.exists(
            data_file_path(filename)
        )

    src_mono_datasets = {}
    for lang_pair in args.mono_lang_pairs:  # lyric-lyric, melody-melody
        lang = lang_pair.split("-")[0]
        if split_exists(split, lang):
            prefix = os.path.join(args.data, "{}.{}".format(split, lang))
        else:
            raise FileNotFoundError(
                "Not Found available {} dataset for ({}) lang".format(split, lang)
            )

        src_mono_datasets[lang_pair] = MMapDataset(prefix)
        print(
            "| monolingual {}-{}: {} examples".format(
                split, lang, len(src_mono_datasets[lang_pair])
            )
        )

    src_para_datasets = {}
    for lang_pair in args.para_lang_pairs:  # lyric-melody
        src, tgt = lang_pair.split("-")
        key = "-".join(sorted([src, tgt]))
        if not split_para_exists(split, key, src):
            raise FileNotFoundError(
                "Not Found available {}-{} para dataset for ({}) lang".format(
                    split, key, src
                )
            )
        if not split_para_exists(split, key, tgt):
            raise FileNotFoundError(
                "Not Found available {}-{} para dataset for ({}) lang".format(
                    split, key, tgt
                )
            )

        prefix = os.path.join(args.data, "{}.{}".format(split, key))
        if "{}.{}".format(key, src) not in src_para_datasets:
            src_para_datasets[key + "." + src] = MMapDataset(prefix + "." + src)
        if "{}.{}".format(key, tgt) not in src_para_datasets:
            src_para_datasets[key + "." + tgt] = MMapDataset(prefix + "." + tgt)

        print(
            "| bilingual {} {}-{}.{}: {} examples".format(
                split, src, tgt, src, len(src_para_datasets[key + "." + src])
            )
        )
        print(
            "| bilingual {} {}-{}.{}: {} examples".format(
                split, src, tgt, tgt, len(src_para_datasets[key + "." + tgt])
            )
        )

    mt_para_dataset = {}
    for lang_pair in args.mt_steps:  # lyric-melody, melody-lyric
        src, tgt = lang_pair.split("-")
        key = "-".join(sorted([src, tgt]))
        src_key = key + "." + src
        tgt_key = key + "." + tgt
        src_dataset = src_para_datasets[src_key]
        tgt_dataset = src_para_datasets[tgt_key]

        mt_para_dataset[lang_pair] = MusicMtDataset(
            src_dataset,
            src_dataset.sizes,
            tgt_dataset,
            tgt_dataset.sizes,
            dicts[src],
            dicts[tgt],
            left_pad_source=args.left_pad_source,
            left_pad_target=args.left_pad_target,
            max_source_positions=args.max_source_positions,
            max_target_positions=args.max_target_positions,
            src_lang=src,
            tgt_lang=tgt,
        )

    eval_para_dataset = {}
    if split != "train":
        for lang_pair in args.valid_lang_pairs:  # lyric-lyric, melody-melody
            src, tgt = lang_pair.split("-")
            if src == tgt:
                src_key = src + "-" + tgt
                tgt_key = src + "-" + tgt
                src_dataset = src_mono_datasets[src_key]
                tgt_dataset = src_mono_datasets[tgt_key]
            else:
                key = "-".join(sorted([src, tgt]))
                src_key = key + "." + src
                tgt_key = key + "." + tgt
                src_dataset = src_para_datasets[src_key]
                tgt_dataset = src_para_datasets[tgt_key]
            eval_para_dataset[lang_pair] = MusicMtDataset(
                src_dataset,
                src_dataset.sizes,
                tgt_dataset,
                tgt_dataset.sizes,
                dicts[src],
                dicts[tgt],
                left_pad_source=args.left_pad_source,
                left_pad_target=args.left_pad_target,
                max_source_positions=args.max_source_positions,
                max_target_positions=args.max_target_positions,
                src_lang=src,
                tgt_lang=tgt,
            )

    mass_mono_datasets = {}
    if split == "train":
        for lang_pair in args.mass_steps:
            src_dataset = src_mono_datasets[lang_pair]
            lang = lang_pair.split("-")[0]

            mass_mono_dataset = MusicMassDataset(
                src_dataset,
                src_dataset.sizes,
                dicts[lang],
                left_pad_source=args.left_pad_source,
                left_pad_target=args.left_pad_target,
                max_source_positions=args.max_source_positions,
                max_target_positions=args.max_target_positions,
                shuffle=True,
                ratio=args.word_mask,
                pred_probs=args.pred_probs,
                lang=lang,
            )
            mass_mono_datasets[lang_pair] = mass_mono_dataset
    return RoundRobinZipDatasets(
        OrderedDict(
            [
                (_get_mt_dataset_key(lang_pair), mt_para_dataset[lang_pair])
                for lang_pair in mt_para_dataset.keys()
            ]
            + [
                (_get_mass_dataset_key(lang_pair), mass_mono_datasets[lang_pair])
                for lang_pair in mass_mono_datasets.keys()
            ]
            + [
                (_get_mt_dataset_key(lang_pair), eval_para_dataset[lang_pair])
                for lang_pair in eval_para_dataset.keys()
            ]
        ),
        eval_key=None if training else args.eval_lang_pair,
    )
