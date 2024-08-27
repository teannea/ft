import asyncio
import json
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, before_sleep_log
import aiofiles
import logging
from tqdm import tqdm
from openai import AsyncOpenAI

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.getLogger("httpx").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

client = AsyncOpenAI()


@dataclass
class EmbeddingResult:
    embedding: List[float]


@retry(
    # stop=stop_after_attempt(6),
    wait=wait_exponential_jitter(initial=1, min=4, max=10),
    before_sleep=before_sleep_log(logger, logging.WARN),
)
async def embed(text: str) -> EmbeddingResult:
    # response = await client.embeddings.create(input=text, model="text-embedding-3-small")
    response = await client.embeddings.create(input=text, model="text-embedding-3-large")

    return EmbeddingResult(embedding=response.data[0].embedding)


async def process_entry(
    index: int, entry: str, semaphore: asyncio.Semaphore
) -> Tuple[int, Optional[EmbeddingResult]]:
    async with semaphore:
        try:
            result = await embed(entry)
            return (index, result)
        except Exception as e:
            logger.error(f"Error processing entry {index}: {entry[:20]}... Error: {e}")
            return (index, None)


async def save_results(results: List[Optional[List[float]]], filename: str):
    async with aiofiles.open(filename, "w") as f:
        await f.write(json.dumps(results, default=str, indent=2))
    logger.info(f"Results saved to {filename}")


async def main():
    max_concurrency = 80  # Adjust this value as needed
    save_interval = 1000  # Save results every 1000 entries

    with open("./extracted-comments.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # data = data[:10]
    total_entries = len(data)
    logger.info(f"Starting processing of {total_entries} entries")

    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = [
        process_entry(i, entry["entry"], semaphore) for i, entry in enumerate(data)
    ]

    results = [None] * total_entries  # Pre-allocate the results list
    progress_bar = tqdm(total=total_entries, desc="Processing entries")

    done_so_far = 0

    for task in asyncio.as_completed(tasks):
        index, result = await task
        results[index] = result.embedding if result else None
        progress_bar.update(1)
        done_so_far += 1

        if (done_so_far + 1) % save_interval == 0:
            await save_results(results, "embedding_results_partial.json")
            logger.info(f"Saved partial results after processing {index + 1} entries")

    progress_bar.close()

    # Save final results
    await save_results(results, "embedding_results_final.json")
    logger.info("Finished processing all entries and saved final results")


if __name__ == "__main__":
    asyncio.run(main())
