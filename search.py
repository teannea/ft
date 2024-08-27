import asyncio
import httpx
import json
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple
from datetime import datetime
import os
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
import aiofiles
import logging
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.getLogger("httpx").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    title: str
    url: str
    description: str
    page_age: Optional[datetime]
    profile: dict
    language: str
    type: str
    subtype: str


@retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    before_sleep=before_sleep_log(logger, logging.WARN),
)
async def search(query: str) -> List[SearchResult]:
    query = query[:128]

    api_key = os.environ.get("BRAVE_API_KEY")
    if not api_key:
        raise ValueError("BRAVE_API_KEY environment variable is not set")

    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }
    params = {"q": query}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

    results = []
    for item in data.get("web", {}).get("results", [])[:8]:  # Limit to first 8 results
        page_age = None
        if "page_age" in item:
            try:
                page_age = datetime.strptime(item["page_age"], "%Y-%m-%dT%H:%M:%S")
            except ValueError:
                pass

        result = SearchResult(
            title=item.get("title", ""),
            url=item.get("url", ""),
            description=item.get("description", ""),
            page_age=page_age,
            profile=item.get("profile", {}),
            language=item.get("language", ""),
            type=item.get("type", ""),
            subtype=item.get("subtype", ""),
        )
        results.append(result)

    return results


async def process_entry(
    index: int, entry: str, semaphore: asyncio.Semaphore
) -> Tuple[int, Optional[List[SearchResult]]]:
    async with semaphore:
        try:
            result = await search(entry)
            return (index, result)
        except Exception as e:
            logger.error(f"Error processing entry {index}: {entry[:20]}... Error: {e}")
            return (index, None)


async def save_results(results: List[Optional[List[SearchResult]]], filename: str):
    serializable_results = [
        [asdict(r) if r else None for r in entry_results] if entry_results else None
        for entry_results in results
    ]
    async with aiofiles.open(filename, "w") as f:
        await f.write(json.dumps(serializable_results, default=str, indent=2))
    logger.info(f"Results saved to {filename}")


async def main():
    max_concurrency = 20  # Adjust this value as needed
    save_interval = 100  # Save results every 10 entries

    with open("./extracted-comments.json", "r", encoding="utf-8") as f:
        data = json.load(f)

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
        results[index] = result
        progress_bar.update(1)
        done_so_far += 1

        if (done_so_far + 1) % save_interval == 0:
            await save_results(results, "search_results_partial.json")
            logger.info(f"Saved partial results after processing {index + 1} entries")

    progress_bar.close()

    # Save final results
    await save_results(results, "search_results_final.json")
    logger.info("Finished processing all entries and saved final results")


if __name__ == "__main__":
    asyncio.run(main())
