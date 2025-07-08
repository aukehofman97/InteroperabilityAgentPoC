
import json
import os
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def clean_json_string(json_string):
    return re.sub(r"```(?:json)?", "", json_string).strip()

def load_prompt_template(path="eventPrompt.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def generate_event_for_record(record, prompt_template):
    prompt = f"{prompt_template}\n\nInput:\n{json.dumps(record, indent=2)}\n\nOutput:"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    raw_content = response.choices[0].message.content
    cleaned_content = clean_json_string(raw_content)
    try:
        return json.loads(cleaned_content)
    except json.JSONDecodeError:
        print(f"❌ Failed to decode event for record: {record}")
        return None
