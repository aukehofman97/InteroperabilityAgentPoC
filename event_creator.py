import json
import os
import re
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# File paths
mapped_data_path = "mapped_data.json"
event_prompt_path = "eventPrompt.txt"
output_events_path = "generated_events.json"

def clean_json_string(json_string):
    # Remove code fences (e.g., ```json ... ```)
    cleaned = re.sub(r"```(?:json)?", "", json_string).strip()
    return cleaned

# Load mapped data
with open(mapped_data_path, "r", encoding="utf-8") as f:
    mapped_data = json.load(f)

# Load event transformation prompt
with open(event_prompt_path, "r", encoding="utf-8") as f:
    event_prompt = f.read()

# Container for all transformed events
all_events = []

# Transform each record
for record in mapped_data[2:3]:
    record_text = json.dumps(record, indent=2)
    prompt = f"{event_prompt}\n\nInput:\n{record_text}\n\nOutput:"

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    print(response.choices[0].message.content)
    print(type(response.choices[0].message.content))
    raw_content = response.choices[0].message.content
    cleaned_content = clean_json_string(raw_content)
    print(cleaned_content)
    try:
        event_json = json.loads(cleaned_content)
        all_events.append(event_json)
    except json.JSONDecodeError:
        print(f"❌ Failed to decode event for record: {record}")
        continue

# Write all events to a JSON file
with open(output_events_path, "w", encoding="utf-8") as f:
    json.dump(all_events, f, indent=2)

print(f"✅ Events saved to {output_events_path}")
