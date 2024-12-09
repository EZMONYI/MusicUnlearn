import argparse
import logging
import os
import sys
from collections import Counter
from multiprocessing import Pool

from utils.dict import Dictionary
from utils.helpers import (
    DatasetBuilder,
    best_fitting_dtype,
    binarize,
    build_ds,
    data_file_path,
    dataset_dest_file,
    dataset_dest_prefix,
    file_name,
    find_offsets,
    index_file_path,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def main(args):
    def make_binary_dataset(vocab, input_prefix, output_prefix, lang, num_workers):
        n_seq_tok = [0, 0]
        replaced = Counter()

        def merge_result(worker_result):
            replaced.update(worker_result["replaced"])
            n_seq_tok[0] += worker_result["nseq"]
            n_seq_tok[1] += worker_result["ntok"]

        input_file = "{}{}".format(
            input_prefix, ("." + lang) if lang is not None else ""
        )
        offsets = find_offsets(input_file, num_workers)
        pool = None
        if num_workers > 1:
            pool = Pool(processes=num_workers - 1)
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                pool.apply_async(
                    build_ds,
                    (
                        args,
                        input_file,
                        vocab,
                        prefix,
                        lang,
                        offsets[worker_id],
                        offsets[worker_id + 1],
                    ),
                    callback=merge_result,
                )
            pool.close()

        ds = DatasetBuilder(
            dataset_dest_file(args, output_prefix, lang, "bin"),
            dtype=best_fitting_dtype(len(vocab)),
        )
        merge_result(
            binarize(
                input_file,
                vocab,
                lambda t: ds.add_item(t),
                offset=0,
                end=offsets[1],
            )
        )
        if num_workers > 1:
            pool.join()
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                temp_file_path = dataset_dest_prefix(args, prefix, lang)
                ds.merge_file_(temp_file_path)
                os.remove(data_file_path(temp_file_path))
                os.remove(index_file_path(temp_file_path))

        ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))

        logger.info(
            "[{}] {}: {} sents, {} tokens, {:.3}% replaced by {}".format(
                lang,
                input_file,
                n_seq_tok[0],
                n_seq_tok[1],
                100 * sum(replaced.values()) / n_seq_tok[1],
                vocab.unk_word,
            )
        )

    def make_all(lang, vocab):
        if args.trainpref:
            make_binary_dataset(vocab, args.trainpref, "train", lang, args.workers)
        if args.validpref:
            make_binary_dataset(vocab, args.validpref, "valid", lang, args.workers)
        if args.unlearnpref:
            make_binary_dataset(vocab, args.unlearnpref, "unlearn", lang, args.workers)

    def dict_path(lang):
        return os.path.join(args.destdir, file_name("dict", lang)) + ".txt"

    def load_dictionary(filename):
        return Dictionary.load(filename)
        os.makedirs(args.destdir, exist_ok=True)

    if args.srcdict:
        src_dict = load_dictionary(args.srcdict)
        src_dict.save(dict_path(args.source_lang))  # save dict.lyric.txt
    if args.tgtdict is not None:
        tgt_dict = load_dictionary(args.tgtdict)
        tgt_dict.save(dict_path(args.target_lang))  # save dict.lyric.txt

    if args.source_lang is not None:
        make_all(args.source_lang, src_dict)

    if args.target_lang is not None:
        make_all(args.target_lang, tgt_dict)


def cli_main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--srcdict")
    parser.add_argument("--tgtdict", default=None)
    parser.add_argument("--trainpref")
    parser.add_argument("--validpref")
    parser.add_argument("--unlearnpref")
    parser.add_argument("--destdir")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--source-lang")
    parser.add_argument("--target-lang", default=None)
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
