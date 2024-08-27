import json
import re

def process_entry(entry):
    pattern = r'[（(]翻[評评]：([^）)]+)[）)]'
    comments = re.findall(pattern, entry)
    cleaned_entry = re.sub(pattern, '', entry)
    return cleaned_entry.strip(), [comment.strip() for comment in comments]

# Read the input file
with open('summary.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Process each item
for item in data:
    if 'entry' in item:
        cleaned_entry, comments = process_entry(item['entry'])
        item['entry'] = cleaned_entry
        item['comment'] = comments

# Write the processed data to a new file
with open('extracted-comments.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("Processing complete. Results written to extracted-comments.json")
