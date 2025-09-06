from datetime import datetime
import google.generativeai as genai
import json
from typing import Optional
from pydantic import BaseModel, Field
import logging

API_KEY = ""

# --------------------------------------------------------------
# SETUP: Configuration and Logging
# --------------------------------------------------------------

# Setting up logging - GOOD! This helps debug the chain
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Configure Gemini - FIXED: genai.configure() doesn't return a client
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro")  # Using Pro for better JSON handling


# --------------------------------------------------------------
# STEP 1: Define Pydantic Models for Each Chain Step
# --------------------------------------------------------------

class EventExtraction(BaseModel):
    """FIRST STEP: Analyze if text describes a calendar event"""
    description: str = Field(description="Raw description of the event")
    is_calendar_event: bool = Field(
        description="Whether this text describes a calendar event"
    )
    confidence_score: float = Field(
        description="Confidence score between 0 and 1", ge=0.0, le=1.0
    )


class EventDetails(BaseModel):
    """SECOND STEP: Extract specific event details"""
    name: str = Field(description="Name of the event")
    date: str = Field(
        description="Date and time of the event in ISO 8601 format"
    )
    duration_minutes: int = Field(
        description="Expected duration in minutes", ge=1
    )
    participants: list[str] = Field(description="List of participants")


class EventConfirmation(BaseModel):
    """THIRD STEP: Generate user confirmation"""
    confirmation_message: str = Field(
        description="Natural language confirmation message"
    )
    calendar_link: Optional[str] = Field(
        description="Generated calendar link if applicable"
    )


# --------------------------------------------------------------
# STEP 2: Helper Functions for Gemini JSON Handling
# --------------------------------------------------------------

def extract_json_from_response(response_text: str) -> dict:
    """
    Extract JSON from Gemini's response text.
    Gemini often wraps JSON in ```json ``` blocks.
    """
    text = response_text.strip()

    # Remove JSON code blocks if present
    if text.startswith("```json"):
        text = text.replace("```json", "").replace("```", "").strip()
    elif text.startswith("```"):
        text = text.replace("```", "").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON from: {text}")
        raise


def create_structured_prompt(
        system_message: str,
        user_message: str,
        response_format: BaseModel
) -> str:
    """
    Create a prompt that asks for specific JSON output.
    This is how we simulate OpenAI's .parse() functionality.
    """
    return f"""
    {system_message}

    You MUST return ONLY valid JSON with this exact structure:
    {response_format.model_json_schema()}

    User input: {user_message}
    """


# --------------------------------------------------------------
# STEP 3: The Prompt Chain Functions
# --------------------------------------------------------------

def extract_event_info(user_input: str) -> EventExtraction:
    """FIRST CHAIN LINK: Analyze if input is a calendar event"""
    logger.info("🔍 Step 1: Analyzing if text describes a calendar event")

    today = datetime.now()
    date_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

    system_message = f"""
    {date_context}
    Analyze if the user's text describes a calendar event. 
    Provide a confidence score based on how clearly it describes an event with time, date, and participants.
    """

    prompt = create_structured_prompt(
        system_message,
        user_input,
        EventExtraction
    )

    response = model.generate_content(prompt)
    json_data = extract_json_from_response(response.text)

    # Convert to Pydantic model for validation
    result = EventExtraction(**json_data)

    logger.info(
        f"✅ Extraction complete - Is event: {result.is_calendar_event}, "
        f"Confidence: {result.confidence_score:.2f}"
    )
    return result


def parse_event_details(description: str) -> EventDetails:
    """SECOND CHAIN LINK: Extract specific event details"""
    logger.info("📋 Step 2: Extracting detailed event information")

    today = datetime.now()
    date_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

    system_message = f"""
    {date_context}
    Extract detailed event information from the description. 
    Convert relative dates (like 'next Tuesday') to absolute dates using today as reference.
    Use ISO 8601 format for the date field.
    """

    prompt = create_structured_prompt(
        system_message,
        description,
        EventDetails
    )

    response = model.generate_content(prompt)
    json_data = extract_json_from_response(response.text)

    result = EventDetails(**json_data)

    logger.info(
        f"✅ Details parsed - Name: {result.name}, "
        f"Date: {result.date}, Duration: {result.duration_minutes}min"
    )
    logger.debug(f"Participants: {', '.join(result.participants)}")
    return result


def generate_confirmation(event_details: EventDetails) -> EventConfirmation:
    """THIRD CHAIN LINK: Generate user confirmation"""
    logger.info("✉️  Step 3: Generating confirmation message")

    system_message = """
    Generate a friendly, natural confirmation message for the event.
    Sign off with your name: Susie
    You may include a calendar link if appropriate.
    """

    # Convert event details to JSON string for the prompt
    details_json = event_details.model_dump_json()

    prompt = create_structured_prompt(
        system_message,
        f"Event details: {details_json}",
        EventConfirmation
    )

    response = model.generate_content(prompt)
    json_data = extract_json_from_response(response.text)

    result = EventConfirmation(**json_data)
    logger.info("✅ Confirmation generated successfully")
    return result


# --------------------------------------------------------------
# STEP 4: The Main Chain Function with Gate Checking
# --------------------------------------------------------------

def process_calendar_request(user_input: str) -> Optional[EventConfirmation]:
    """
    MAIN CHAIN: Orchestrate the prompt chaining process

    This is like a factory assembly line:
    1. Quality Check → 2. Detail Extraction → 3. Packaging
    """
    logger.info("🚀 Starting calendar request processing chain")

    try:
        # 🔗 FIRST LINK: Extraction and gate check
        extraction = extract_event_info(user_input)

        # 🚧 GATE CHECK: Only proceed if this is likely a calendar event
        if not extraction.is_calendar_event or extraction.confidence_score < 0.6:
            logger.warning(
                f"❌ Gate check failed - Not a calendar event "
                f"(confidence: {extraction.confidence_score:.2f})"
            )
            return None

        logger.info("✅ Gate check passed - Proceeding with event processing")

        # 🔗 SECOND LINK: Extract detailed information
        event_details = parse_event_details(extraction.description)

        # 🔗 THIRD LINK: Generate user confirmation
        confirmation = generate_confirmation(event_details)

        logger.info("🎉 Prompt chain completed successfully!")
        return confirmation

    except Exception as e:
        logger.error(f"💥 Chain processing failed: {str(e)}")
        return None


# --------------------------------------------------------------
# STEP 5: Test the Prompt Chain
# --------------------------------------------------------------

def test_prompt_chain():
    """Test our prompt chain with different inputs"""

    test_cases = [
        {
            "input": "Let's schedule a 1h team meeting next Tuesday at 2pm with Alice and Bob to discuss the project roadmap.",
            "description": "Valid calendar event with relative date"
        },
        {
            "input": "Coffee chat with Sarah tomorrow at 3pm for 30 minutes",
            "description": "Valid informal event"
        },
        {
            "input": "Can you send an email to the team about the project?",
            "description": "Not a calendar event - should fail gate check"
        }
    ]

    print("🧪 TESTING PROMPT CHAINING")
    print("=" * 60)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['description']}")
        print(f"   Input: '{test_case['input']}'")
        print("   " + "-" * 50)

        result = process_calendar_request(test_case["input"])

        if result:
            print(f"   ✅ Success: {result.confirmation_message}")
            if result.calendar_link:
                print(f"   📅 Calendar link: {result.calendar_link}")
        else:
            print("   ❌ Not a calendar event (gate check failed)")

        if i < len(test_cases):
            print("\n" + "=" * 60)


# --------------------------------------------------------------
# RUN THE DEMONSTRATION
# --------------------------------------------------------------

if __name__ == "__main__":
    # Test the prompt chain
    test_prompt_chain()

    # Show the chain structure
    print("\n" + "=" * 60)
    print("🔗 PROMPT CHAIN STRUCTURE:")
    print("=" * 60)
    print("""
    User Input
    │
    ▼
    [Step 1] EventExtraction → "Is this a calendar event?"
    │
    ▼
    [Gate Check] → Confidence > 0.6? → If not: STOP ❌
    │
    ▼  
    [Step 2] EventDetails → "Extract name, date, participants"
    │
    ▼
    [Step 3] EventConfirmation → "Generate friendly message"
    │
    ▼
    Final Result ✅
    """)