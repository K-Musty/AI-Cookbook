from typing import Optional, Literal, List
from pydantic import BaseModel, Field, validator
import google.generativeai as genai
import json
import logging
from datetime import datetime

API_KEY = ''

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro")


# --------------------------------------------------------------
# Step 1: Define Pydantic Models for Routing and Responses
# --------------------------------------------------------------

class CalendarRequestType(BaseModel):
    """Router LLM call: Determine the type of calendar request"""
    request_type: Literal["new_event", "modify_event", "other"] = Field(
        description="Type of calendar request being made"
    )
    confidence_score: float = Field(
        description="Confidence score between 0 and 1",
        ge=0.0,
        le=1.0
    )
    description: str = Field(description="Cleaned description of the request")

    @validator('confidence_score')
    def validate_confidence(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Confidence score must be between 0 and 1')
        return round(v, 2)


class Change(BaseModel):
    """Details for changing an existing event"""
    field: str = Field(description="Field to change")
    new_value: str = Field(description="New value for the field")


class NewEventDetails(BaseModel):
    """Details for creating a new event"""
    name: str = Field(description="Name of the event")
    date: str = Field(description="Date and time of the event (ISO 8601)")
    duration_minutes: int = Field(
        description="Duration in minutes",
        ge=1
    )
    participants: List[str] = Field(description="List of participants")


class ModifyEventDetails(BaseModel):
    """Details for modifying an existing event"""
    event_identifier: str = Field(
        description="Description to identify the existing event"
    )
    changes: List[Change] = Field(description="List of changes to make")
    participants_to_add: List[str] = Field(
        default_factory=list,
        description="New participants to add"
    )
    participants_to_remove: List[str] = Field(
        default_factory=list,
        description="Participants to remove"
    )


class CalendarResponse(BaseModel):
    """Final response format"""
    success: bool = Field(description="Whether the operation was successful")
    message: str = Field(description="User-friendly response message")
    calendar_link: Optional[str] = Field(
        default=None,
        description="Calendar link if applicable"
    )


# --------------------------------------------------------------
# Step 2: Helper Functions for Gemini + Pydantic Integration
# --------------------------------------------------------------

def gemini_parse_response(response_text: str, response_model: BaseModel):
    """
    Extract JSON from Gemini response and parse with Pydantic

    This is our equivalent of OpenAI's .parse() method
    """
    text = response_text.strip()

    # Clean JSON markdown blocks
    if text.startswith("```json"):
        text = text.replace("```json", "").replace("```", "").strip()

    try:
        json_data = json.loads(text)
        return response_model(**json_data)
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to parse response: {e}")
        logger.error(f"Raw response: {text}")
        raise


def create_structured_prompt(
        system_message: str,
        user_input: str,
        response_model: BaseModel
) -> str:
    """
    Create a prompt that asks for specific JSON output matching Pydantic model
    """
    schema = response_model.model_json_schema()

    return f"""
    {system_message}

    You MUST return ONLY valid JSON with this exact structure:
    {json.dumps(schema, indent=2)}

    User input: "{user_input}"
    """


# --------------------------------------------------------------
# Step 3: The Router Function
# --------------------------------------------------------------

def route_calendar_request(user_input: str) -> CalendarRequestType:
    """Router LLM call to determine the type of calendar request"""
    logger.info("🔄 Routing calendar request")

    system_message = """
    Determine if this is a request to create a new calendar event or modify an existing one.
    Return "new_event" for scheduling new meetings, "modify_event" for changing existing ones,
    or "other" for non-calendar requests.
    """

    prompt = create_structured_prompt(system_message, user_input, CalendarRequestType)
    response = model.generate_content(prompt)

    result = gemini_parse_response(response.text, CalendarRequestType)

    logger.info(
        f"✅ Request routed as: {result.request_type} "
        f"(confidence: {result.confidence_score:.2f})"
    )
    return result


# --------------------------------------------------------------
# Step 4: Handler Functions - The Specialists
# --------------------------------------------------------------

def handle_new_event(description: str) -> CalendarResponse:
    """Process a new event request"""
    logger.info("📅 Processing new event request")

    today = datetime.now().strftime("%A, %B %d, %Y")

    system_message = f"""
    Today is {today}. Extract details for creating a new calendar event.
    Convert relative dates (like "next Tuesday") to absolute dates using today as reference.
    Use ISO 8601 format for the date field.
    """

    prompt = create_structured_prompt(system_message, description, NewEventDetails)
    response = model.generate_content(prompt)

    details = gemini_parse_response(response.text, NewEventDetails)

    logger.info(f"✅ New event: {details.name} at {details.date}")
    logger.debug(f"Participants: {', '.join(details.participants)}")

    return CalendarResponse(
        success=True,
        message=f"Created new event '{details.name}' for {details.date} "
                f"with {', '.join(details.participants)}",
        calendar_link=f"calendar://new?event={details.name}"
    )


def handle_modify_event(description: str) -> CalendarResponse:
    """Process an event modification request"""
    logger.info("🔄 Processing event modification request")

    system_message = """
    Extract details for modifying an existing calendar event.
    Identify which event is being discussed and what changes are requested.
    """

    prompt = create_structured_prompt(system_message, description, ModifyEventDetails)
    response = model.generate_content(prompt)

    details = gemini_parse_response(response.text, ModifyEventDetails)

    logger.info(f"✅ Modifying event: {details.event_identifier}")
    logger.debug(f"Changes: {len(details.changes)} modifications")

    change_descriptions = [f"{change.field} to {change.new_value}" for change in details.changes]
    changes_text = ", ".join(change_descriptions)

    return CalendarResponse(
        success=True,
        message=f"Modified event '{details.event_identifier}'. Changes: {changes_text}",
        calendar_link=f"calendar://modify?event={details.event_identifier}"
    )


# --------------------------------------------------------------
# Step 5: The Main Routing Workflow
# --------------------------------------------------------------

def process_calendar_request(user_input: str) -> Optional[CalendarResponse]:
    """Main function implementing the routing workflow"""
    logger.info("🚀 Processing calendar request")

    try:
        # Step 1: Route the request
        route_result = route_calendar_request(user_input)

        # Step 2: Gate check - confidence threshold
        if route_result.confidence_score < 0.7:
            logger.warning(f"❌ Low confidence score: {route_result.confidence_score:.2f}")
            return None

        # Step 3: Route to appropriate handler
        if route_result.request_type == "new_event":
            return handle_new_event(route_result.description)
        elif route_result.request_type == "modify_event":
            return handle_modify_event(route_result.description)
        else:
            logger.warning("❌ Request type not supported")
            return None

    except Exception as e:
        logger.error(f"💥 Routing workflow failed: {e}")
        return CalendarResponse(
            success=False,
            message="Sorry, I encountered an error processing your calendar request."
        )


# --------------------------------------------------------------
# Step 6: Test Functions with Pydantic Models
# --------------------------------------------------------------

def test_routing_system():
    """Test our routing system with different types of requests"""

    test_cases = [
        {
            "input": "Let's schedule a team meeting next Tuesday at 2pm with Alice and Bob",
            "expected_type": "new_event",
            "description": "New event with relative date"
        },
        {
            "input": "Can you move the team meeting with Alice and Bob to Wednesday at 3pm instead?",
            "expected_type": "modify_event",
            "description": "Modify existing event"
        },
        {
            "input": "What's the weather like today?",
            "expected_type": "other",
            "description": "Not a calendar request"
        }
    ]

    print("🧪 TESTING ROUTING SYSTEM WITH PYDANTIC")
    print("=" * 60)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['description']}")
        print(f"   Input: '{test_case['input']}'")
        print("   " + "-" * 50)

        result = process_calendar_request(test_case["input"])

        if result:
            print(f"   ✅ Success: {result.message}")
            if result.calendar_link:
                print(f"   📅 Link: {result.calendar_link}")

            # Show the structured data
            print(f"   🏗️  Structured response: {result.model_dump()}")
        else:
            print("   ❌ Request not handled (routing failed or low confidence)")

        if i < len(test_cases):
            print("\n" + "=" * 60)


# --------------------------------------------------------------
# Step 7: Demonstrate Pydantic Benefits
# --------------------------------------------------------------

def demonstrate_pydantic_benefits():
    """Show the benefits of using Pydantic models"""

    print("\n" + "=" * 60)
    print("🎯 PYDANTIC BENEFITS IN ROUTING SYSTEMS")
    print("=" * 60)

    benefits = """
    Why Pydantic is better than dictionaries for routing:

    1. ✅ TYPE SAFETY: Compile-time type checking
    2. ✅ VALIDATION: Automatic data validation
    3. ✅ DOCUMENTATION: Self-documenting models
    4. ✅ SERIALIZATION: Easy JSON conversion
    5. ✅ DEFAULT VALUES: Sensible defaults
    6. ✅ ERROR HANDLING: Clear validation errors

    Example of Pydantic validation:
    • confidence_score must be between 0-1
    • duration_minutes must be positive
    • Required fields are enforced
    • Literal types restrict to specific values
    """

    print(benefits)

    # Show example validation
    print("\n🔍 PYDANTIC VALIDATION EXAMPLE:")
    try:
        # This will fail validation
        bad_data = {
            "request_type": "invalid_type",  # Not in Literal
            "confidence_score": 1.5,  # Out of range
            "description": "test"
        }
        test_model = CalendarRequestType(**bad_data)
        print("   ❌ Should have failed validation!")
    except Exception as e:
        print(f"   ✅ Correctly failed: {e}")


# --------------------------------------------------------------
# Run the Demonstration
# --------------------------------------------------------------

if __name__ == "__main__":
    # Test the routing system with Pydantic
    test_routing_system()

    # Demonstrate Pydantic benefits
    demonstrate_pydantic_benefits()

    print("\n" + "=" * 60)
    print("🎯 KEY PYDANTIC FEATURES USED:")
    print("=" * 60)
    print("1. Field validation with ge/le constraints")
    print("2. Literal types for restricted values")
    print("3. Custom validators with @validator")
    print("4. Default values and default_factory")
    print("5. Model serialization with model_dump()")
    print("6. JSON schema generation with model_json_schema()")