#!/bin/env python


import gzip
import json
import logging
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Generator, Iterable, List

import click
import torch
from sentence_transformers import SentenceTransformer, util

Batch = tuple[int, Iterable]
Passage = List[str]


def get_logger() -> logging.Logger:
    logger = logging.getLogger("simple-wiki-job")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] [%(funcName)s]: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = get_logger()


def get_data_in_x_batches(data: Iterable, num_batches: int = 1) -> Generator:
    """Split `data` to `num_batches`"""

    batch_size = math.ceil(len(data) / num_batches)
    batch_id = 0
    for i in range(0, len(data), batch_size):
        yield (batch_id, data[i : i + batch_size])
        batch_id += 1


def process_batch(one_model_per_gpu: list):
    """Encode on defined device"""

    def _process_batch(payload: Batch):
        batch_id, data = payload
        device = f"cuda:{batch_id}"
        return one_model_per_gpu[batch_id].encode(
            data,
            convert_to_tensor=True,
            device=device,
        )

    return _process_batch


def get_simple_wikipedia_path(wikipedia_filepath: str) -> str:
    """Download prepared Simple English Wikipedia and return its path.

    As dataset, we use Simple English Wikipedia.
    Compared to the full English wikipedia, it has only
    about 170k articles. We split these articles into
    paragraphs and encode them with the bi-encoder.
    """

    if not os.path.exists(wikipedia_filepath):
        logger.info("Simple English Wikipedia not found locally. Downloading.")
        util.http_get(
            "http://sbert.net/datasets/simplewiki-2020-11-01.jsonl.gz",
            wikipedia_filepath,
        )
    return wikipedia_filepath


def get_passages(wikipedia_filepath: str) -> List[Passage]:
    passages = []
    with gzip.open(wikipedia_filepath, "rt", encoding="utf8") as fIn:
        for line in fIn:
            data = json.loads(line.strip())
            for paragraph in data["paragraphs"]:
                # We encode the passages as [title, text]
                passages.append([data["title"], paragraph])

    logger.info(f"Number of passages: {len(passages)}")
    return passages


def persist_transformed_passages(passages: list, passages_filepath: str):
    try:
        logger.info(f"Persisting passages to {passages_filepath}")
        with gzip.open(passages_filepath, "wb") as f:
            d = json.dumps(
                [
                    {
                        "corpus_id": i,
                        "corpus_title": passages[i][0],
                        "corpus_text": passages[i][1],
                    }
                    for i in range(len(passages))
                ],
            )
            f.write(d.encode("utf-8"))
    except Exception as ex:
        logger.warning(f"Failed to persist passages to {passages_filepath}")


def info_on_gpu_setup():
    if not torch.cuda.is_available():
        logger.warning("No GPU found. Please add GPU to your setup.")
    else:
        no_of_gpus = torch.cuda.device_count()
        logger.info("CUDA found. Available devices:")
        for i in range(no_of_gpus):
            logger.info(f"Device {i}: {torch.cuda.get_device_name(i)}")


def get_model_per_gpu(
    cache_folder: str, model_name: str, no_of_gpus: int
) -> List[SentenceTransformer]:
    return [
        SentenceTransformer(
            model_name,
            cache_folder=cache_folder,
            device=f"cuda:{i}",
        )
        for i in range(no_of_gpus)
    ]


def save_single_pt_file(
    model_name: str, embeddings_dir: str, pt_file_prefix: str, results: List
):
    try:
        model_name = model_name.replace("/", "_")
        trg = os.path.join(embeddings_dir, f"{pt_file_prefix}--{model_name}.pt")
        cated = torch.cat([results[i].to(f"cuda:0") for i in range(len(results))])

        torch.save(cated, trg)
    except Exception as ex:
        logger.error(
            f"Failed to save embedding as single file {trg=}. Try again as"
            f" separate files. Error we got: {ex}"
        )


def save_multiple_pt_files(
    model_name: str,
    embeddings_dir: str,
    pt_file_prefix: str,
    results: List,
    no_of_gpus: int,
):
    try:
        for i in range(no_of_gpus):
            trg = os.path.join(embeddings_dir, f"{pt_file_prefix}--{model_name}.{i}.pt")
            logger.info(f"Saving cuda:{i} to {trg=}")
            torch.save(results[i], trg)
    except Exception as ex:
        logger.error(
            f"Failed to save embedding as a separate file {trg=}. Error we got: {ex}"
        )


@click.command()
@click.option(
    "--model-name",
    default="nq-distilbert-base-v1",
    help="Model that should be used to create embeddings.",
)
@click.option(
    "--model-cache-dir",
    default="/root/models",
    help="Root directory for model cache.",
)
@click.option(
    "--data-dir",
    default="/root/data",
    help="Directory to save workfiles.",
)
@click.option(
    "--embeddings-dir",
    default="/root/embeddings",
    help="Directory to save embeddings.",
)
@click.option(
    "--no-of-gpus",
    default=1,
    help=(
        "Number of GPUs to use. Make sure it is a correct number, as it is not checked"
        " programmatically."
    ),
)
@click.option(
    "--pt-file-prefix",
    required=True,
    help=(
        "Prefix to add to serialized pt file. Allows to create multiple embeddings and"
        " simple, name-based versionning."
    ),
)
@click.option(
    "--save-single-pt",
    is_flag=True,
    help=(
        "If set, all embeddings will be put to cuda:0 and saved as signle file."
        " Otherwise one file per GPU will be saved."
    ),
)
def run_job(
    model_name: str,
    model_cache_dir: str,
    data_dir: str,
    embeddings_dir: str,
    no_of_gpus: int,
    pt_file_prefix: str,
    save_single_pt: bool,
):
    logger.info("Job started")
    logger.info(
        f"Using parameters: {model_name=}, {model_cache_dir=}, {data_dir=},"
        f" {embeddings_dir=}, {no_of_gpus=}, {pt_file_prefix=}, {save_single_pt=}"
    )

    model_cache_folder = os.path.join(model_cache_dir, model_name)

    os.makedirs(model_cache_folder, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)

    wikipedia_filepath = os.path.join(data_dir, "simplewiki-2020-11-01.jsonl.gz")
    passages_filepath = os.path.join(data_dir, "passages-2020-11-01.json.gz")

    # just diplay info, no action is taken if no GPUs found.
    info_on_gpu_setup()

    get_simple_wikipedia_path(wikipedia_filepath)
    passages = get_passages(wikipedia_filepath)

    persist_transformed_passages(passages, passages_filepath)

    one_model_per_gpu = get_model_per_gpu(model_cache_folder, model_name, no_of_gpus)

    data_batches = get_data_in_x_batches(passages, num_batches=no_of_gpus)

    process_batch_with_model = process_batch(one_model_per_gpu)

    with ThreadPoolExecutor(max_workers=no_of_gpus) as executor:
        results = list(executor.map(process_batch_with_model, data_batches))

    if save_single_pt:
        logger.info("Trying to put all embeddings to one GPU.")
        save_single_pt_file(model_name, embeddings_dir, pt_file_prefix, results)
    else:
        logger.info("Saving one file per GPU.")
        save_multiple_pt_files(
            model_name, embeddings_dir, pt_file_prefix, results, no_of_gpus
        )

    logger.info("Job finished")


if __name__ == "__main__":
    run_job()
