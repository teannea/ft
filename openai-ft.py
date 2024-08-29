import duckdb
import json

# Connect to the database
con = duckdb.connect('data.db')

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
with open('finetune-data.jsonl', 'w') as f:
    for row in results:
        f.write(json.dumps(row[0]) + '\n')

con.close()
