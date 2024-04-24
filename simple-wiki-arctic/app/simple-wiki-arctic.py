import logging
import os
import time
from typing import List

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from typing_extensions import TypedDict

logging.config.fileConfig("logging.conf", disable_existing_loggers=False)
logger = logging.getLogger(__name__)


# search types
class Hit(TypedDict):
    corpus_id: int
    score: float


Hits = tuple[int, tuple[Hit]]


class AllHits(BaseModel):
    data: List[Hits]


# payload types
SingleEntry = tuple[int, str]


class Payload(BaseModel):
    data: List[SingleEntry]


# embedding types
EmbeddedEntry = tuple[int, List[float]]


class Embedding(BaseModel):
    data: List[EmbeddedEntry]


# default route type
class OK(BaseModel):
    status: str


# our application
app = FastAPI()

# add cors
origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# initialization of the model
model_name = "Snowflake/snowflake-arctic-embed-l"
model_path = f"/root/models/{model_name}"
os.makedirs(model_path, exist_ok=True)
model = SentenceTransformer(model_name, cache_folder=model_path)

# initializaiton of the embeddings
embeddings_path = f"/root/data/embeddings"
embeddings_file_name = "ver1--Snowflake_snowflake-arctic-embed-l.pt"
embeddings_file_path = os.path.join(embeddings_path, embeddings_file_name)
try:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    logger.info(f"Loading embeddings to {device}")
    corpus_embeddings = torch.load(
        embeddings_file_path, map_location=torch.device(device)
    ).float()
except Exception as ex:
    logger.critical(
        f"Can't load embeddings from {embeddings_file_path}, got an exception: {ex}.\n"
        " Exiting."
    )
    exit(1)


def info_on_gpu_setup():
    if not torch.cuda.is_available():
        logger.warning("No GPU found. Please add GPU to your setup.")
    else:
        no_of_gpus = torch.cuda.device_count()
        logger.info("CUDA found. Available devices:")
        for i in range(no_of_gpus):
            logger.info(f"Device {i}: {torch.cuda.get_device_name(i)}")


@app.get("/")
async def root():
    logger.info("/ was requested")
    return OK(status="All good. Go to /docs for documentation")


@app.post("/search")
async def search(payload: Payload) -> AllHits:
    """Search for similar documents"""

    top_k = 5

    logger.info(f"/search was requested with {len(payload.data)} elements")
    all_hits = AllHits(data=[])

    _st = time.time()

    text_ids, texts = zip(*payload.data)

    question_embedding = model.encode(
        texts, convert_to_tensor=True, show_progress_bar=False
    )
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)

    all_hits.data = list(zip(text_ids, [tuple(hit) for hit in hits]))

    logger.info("/search processing time: %f [ms]", time.time() - _st)

    return all_hits


@app.post("/embed")
async def embed(payload: Payload) -> Embedding:
    """Create embedding for strings."""

    logger.info(f"/embed was requested with {len(payload.data)} elements")

    embedding = Embedding(data=[])

    _st = time.time()
    text_ids, texts = zip(*payload.data)

    embedded_texts = model.encode(texts, show_progress_bar=False).tolist()
    all_embeddings = list(zip(text_ids, embedded_texts))

    logger.info("/embed processing time: %f [ms]", time.time() - _st)
    logger.info(f"{len(embedded_texts[0])=}")

    embedding.data = all_embeddings
    return embedding


info_on_gpu_setup()
