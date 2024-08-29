import duckdb
import json

# Connect to the database
con = duckdb.connect("data.db")

# Query to join tables and format data
query = """
SELECT
    {
        'messages': [
            {'role': 'user', 'content': ne.expanded_content},
            {'role': 'assistant', 'content': n.content}
        ]
    } AS data
FROM
    news_expanded ne
JOIN
    news n ON ne.news_id = n.id
"""

# Execute query and fetch results
results = con.execute(query).fetchall()

# Write to JSONL file
with open("finetune-data-1k-system.jsonl", "w") as f:
    for row in results[-1000:]:
        example = row[0]

        example['messages'].insert(
            0,
            {
                "role": "system",
                "content": "请将以下新闻总结为一条 200 字左右的新闻快报。请只保留最重要的信息。",
            },
        )
        # print(example)
        # exit(1)
        f.write(json.dumps(example) + '\n')

con.close()
