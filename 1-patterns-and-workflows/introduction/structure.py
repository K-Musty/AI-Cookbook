import google.generativeai as genai
from pydantic import BaseModel
import json

API_KEY = ""


genai.configure(api_key=API_KEY)

class CalenderEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

# Helper Function to remove and clean unsupported fields i.e title etc
def get_gemini_schema(pydantic_model):
    raw_schema = pydantic_model.model_json_schema()

    # Clean up the schema to be compatible with Gemini
    gemini_schema = {
        "type": raw_schema.get("type", "OBJECT"),
        "properties": raw_schema.get("properties", {}),
        "required": raw_schema.get("required", [])
    }

    # Clean each property to remove 'title' and other unsupported fields
    for prop_name, prop_details in gemini_schema["properties"].items():
        if "title" in prop_details:
            del prop_details["title"]
        # Handle array items if they have titles
        if prop_details.get("type") == "array" and "items" in prop_details:
            if "title" in prop_details["items"]:
                del prop_details["items"]["title"]

    return gemini_schema


model = genai.GenerativeModel('gemini-1.5-flash')


prompt = "Abdulrahman and Abdullahi are going to a science fair this Friday."

response = model.generate_content(
    [prompt],
    generation_config=genai.types.GenerationConfig(
        response_mime_type="application/json",
        response_schema=get_gemini_schema(CalenderEvent)
    )
)


response_json = json.loads(response.text)
event = CalenderEvent(**response_json)

print(f"Event Name: {event.name}")
print(f"Event Date: {event.date}")
print(f"Participants: {', '.join(event.participants)}")
