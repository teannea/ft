import asyncio
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    wait_exponential_jitter,
    retry_if_exception_type,
    before_sleep_log,
)
import httpx
import json
import duckdb
from tqdm import tqdm
import logging
from openai import AsyncOpenAI

logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log format
    datefmt="%Y-%m-%d %H:%M:%S",  # Date format
    handlers=[
        logging.FileHandler("db.log"),  # Log to a file
        logging.StreamHandler(),  # Also log to console
    ],
)
logger = logging.getLogger("db")
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)


def init(con):
    con.sql("CREATE SEQUENCE IF NOT EXISTS news_id START 1")
    con.sql("CREATE SEQUENCE IF NOT EXISTS source_id START 1")

    con.sql("""
    CREATE TABLE IF NOT EXISTS news (
        id INTEGER PRIMARY KEY DEFAULT NEXTVAL('news_id'),
        url VARCHAR,
        content VARCHAR,
        comment VARCHAR,
        date DATE
    )
    """)

    con.sql("""
    CREATE TABLE IF NOT EXISTS sources (
        id INTEGER PRIMARY KEY DEFAULT NEXTVAL('source_id'),
        news_id INTEGER,
        url VARCHAR,
        title VARCHAR,
        description VARCHAR,
        page_age DATETIME,
        language VARCHAR,
        content VARCHAR,
        topk INTEGER,
        FOREIGN KEY (news_id) REFERENCES news(id)
    )
    """)

    with open("./extracted-comments.json", "r", encoding="utf-8") as file:
        news = json.load(file)

    with open("./search_results_final.json", "r", encoding="utf-8") as file:
        searches = json.load(file)

    for i, (entry, sources) in tqdm(enumerate(zip(news, searches))):
        con.execute(
            "INSERT INTO news (id, url, content, comment, date) VALUES (?, ?, ?, ?, ?)",
            (
                i,
                entry["url"],
                entry["entry"],
                entry["comment"][0] if entry["comment"] else None,
                entry["datetime"],
            ),
        )

        if not sources:
            continue

        for topk, source in enumerate(sources):
            con.execute(
                "INSERT INTO sources (news_id, url, title, description, page_age, language, topk) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    i,
                    source["url"],
                    source["title"],
                    source["description"],
                    source["page_age"],
                    source["language"],
                    topk,
                ),
            )

    con.commit()


def init_embed(con):
    con.sql("""
    CREATE TABLE IF NOT EXISTS embeddings (
        news_id INTEGER PRIMARY KEY,
        embed FLOAT[3072],
        FOREIGN KEY (news_id) REFERENCES news(id)
    )
    """)
    con.sql("INSTALL vss")
    con.sql("LOAD vss")
    con.sql("SET hnsw_enable_experimental_persistence = true")
    con.sql("CREATE INDEX idx ON embeddings USING HNSW (embed)")

    with open("./embedding_results_final.json", "r", encoding="utf-8") as file:
        embeddings = json.load(file)

    for i, embed in tqdm(enumerate(embeddings)):
        con.execute(
            "INSERT INTO embeddings (news_id, embed) VALUES (?, ?)",
            (
                i,
                embed,
            ),
        )


@retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=(
        retry_if_exception_type(httpx.HTTPError)
        | retry_if_exception_type(httpx.RequestError)
    ),
    before_sleep=before_sleep_log(logger, logging.WARN),
)
async def crawl_url(client, url):
    logger.info(f"Crawling URL: {url}")
    paper_url = f"https://r.jina.ai/{url}"
    response = await client.get(paper_url)
    response.raise_for_status()
    return response.text


async def process_url(semaphore, client, row_id, url, con):
    async with semaphore:
        try:
            result = await crawl_url(client, url)
            con.execute("UPDATE sources SET content = ? WHERE id = ?", (result, row_id))
            logger.info(f"Successfully processed row {row_id}")
        except Exception as e:
            logger.error(f"Error crawling URL for row {row_id}: {str(e)}")


async def update_source_content(concurrency=10):
    con = duckdb.connect("data.db")

    # Fetch rows with NULL content
    rows = con.execute("SELECT id, url FROM sources WHERE content IS NULL").fetchall()

    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=30) as client:
        tasks = []
        for row in tqdm(rows, desc="Preparing tasks"):
            task = asyncio.create_task(
                process_url(semaphore, client, row[0], row[1], con)
            )
            tasks.append(task)

        # Wait for all tasks to complete
        for task in tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="Processing URLs"
        ):
            await task

        con.commit()

    con.close()


@retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=(
        retry_if_exception_type(httpx.HTTPError)
        | retry_if_exception_type(httpx.RequestError)
    ),
)
def crawl_url_sync(url):
    paper_url = f"https://r.jina.ai/{url}"
    logger.info(f"Crawling: {paper_url}")  # Debug print

    with httpx.Client(timeout=30, verify=False) as client:
        try:
            response = client.get(paper_url, follow_redirects=True)
            response.raise_for_status()
            print(f"Successfully crawled: {paper_url}")  # Debug print
            return response.text
        except httpx.HTTPError as e:
            print(f"HTTP error occurred while crawling {paper_url}: {str(e)}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred while crawling {paper_url}: {str(e)}")
            raise


client = AsyncOpenAI()


@retry(
    # stop=stop_after_attempt(6),
    wait=wait_exponential_jitter(initial=1, max=10),
    before_sleep=before_sleep_log(logger, logging.WARN),
)
async def expand_content(content):
    response = await client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "请将以下新闻总结扩展为一篇500字到1000字新闻报道，包括更多细节和背景信息。新增的内容应该是合理的，但不必是真实的。",
            },
            {"role": "user", "content": content},
        ],
        model="gpt-4o",
    )
    print(response)
    return response.choices[0].message.content


async def expand_task(id, content, semaphore):
    async with semaphore:
        try:
            result = await expand_content(content)
            return id, result
        except Exception as e:
            logger.error(f"Error expanding content: {str(e)}")
            return None


async def expand(con, concurrency=100):
    con.sql("CREATE SEQUENCE IF NOT EXISTS expand_id START 1")
    con.sql("""CREATE TABLE IF NOT EXISTS news_expanded (
        id INTEGER PRIMARY KEY DEFAULT NEXTVAL('expand_id'),
        news_id INTEGER,
        expanded_content VARCHAR,
        FOREIGN KEY (news_id) REFERENCES news(id)
    )""")

    rows = con.execute("SELECT id, content FROM news").fetchall()

    semaphore = asyncio.Semaphore(concurrency)

    tasks = []
    for row in tqdm(rows, desc="Preparing tasks"):
        task = asyncio.create_task(expand_task(row[0], row[1], semaphore))
        tasks.append(task)

    progress_bar = tqdm(total=len(tasks), desc="Expanding content")
    for task in asyncio.as_completed(tasks):
        news_id, content = await task
        print(content)
        con.execute(
            "INSERT INTO news_expanded (news_id, expanded_content) VALUES (?, ?)",
            (news_id, content),
        )
        progress_bar.update(1)
    progress_bar.close()


if __name__ == "__main__":
    con = duckdb.connect("data.db")
    # # init(con)
    # init_embed(con)
    # asyncio.run(update_source_content(concurrency=100))
    # print(crawl_url_sync("https://google.com"))
    asyncio.run(expand(con, concurrency=300))
    con.close()
