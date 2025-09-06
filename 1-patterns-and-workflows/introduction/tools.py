import json
import requests
import google.generativeai as genai
from pydantic import BaseModel, Field

API_KEY = ""

# Configure Gemini
genai.configure(api_key=API_KEY)


# --------------------------------------------------------------
# Simulated weather function (to avoid API blocking)
# --------------------------------------------------------------

def get_weather(latitude: float, longitude: float) -> dict[str, any]:
    """Simulate weather data for given coordinates."""
    # Simulate different weather based on coordinates
    weather_scenarios = [
        {"temperature_2m": 22.5, "wind_speed_10m": 15.0, "conditions": "sunny"},
        {"temperature_2m": 18.2, "wind_speed_10m": 8.5, "conditions": "partly_cloudy"},
        {"temperature_2m": 12.7, "wind_speed_10m": 22.0, "conditions": "rainy"},
        {"temperature_2m": 28.9, "wind_speed_10m": 5.0, "conditions": "clear"}
    ]

    # Pick a scenario based on coordinates (simple hash)
    scenario_index = hash(f"{latitude}_{longitude}") % len(weather_scenarios)
    weather_data = weather_scenarios[scenario_index]

    print(f"🌤️  Simulated weather for ({latitude}, {longitude}): {weather_data}")
    return weather_data


# --------------------------------------------------------------
# Define the tool for Gemini
# --------------------------------------------------------------

weather_tool = {
    "function_declarations": [
        {
            "name": "get_weather",
            "description": "Get current weather data including temperature in celsius and wind speed for provided coordinates.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "latitude": {
                        "type": "number",
                        "description": "Latitude coordinate"
                    },
                    "longitude": {
                        "type": "number",
                        "description": "Longitude coordinate"
                    },
                },
                "required": ["latitude", "longitude"]
            }
        }
    ]
}


# --------------------------------------------------------------
# Manual function calling (more reliable than automatic)
# --------------------------------------------------------------

def run_weather_assistant():
    """Main function to run the weather assistant with manual tool handling."""

    # Create model
    model = genai.GenerativeModel(
        'gemini-1.5-flash',
        tools=[weather_tool],
        system_instruction="You are a helpful weather assistant. Use the get_weather tool when users ask about weather and provide coordinates."
    )

    # Start chat WITHOUT automatic calling
    chat = model.start_chat(enable_automatic_function_calling=False)

    # User message with explicit coordinates
    user_message = "What is the current weather at latitude 48.8566 and longitude 2.3522?"

    print(f"👤 User: {user_message}")

    try:
        # Step 1: Send initial message
        response = chat.send_message(user_message)

        # Step 2: Check if model wants to call a function
        should_call_function = False
        function_name = None
        function_args = None

        # Safely inspect the response parts
        if response.parts:
            for part in response.parts:
                # Check for function call in a safe way
                if hasattr(part, 'function_call') and part.function_call:
                    should_call_function = True
                    function_name = part.function_call.name
                    function_args = part.function_call.args
                    print(f"🔧 Model wants to call: {function_name} with args: {function_args}")
                    break

        # Step 3: If function should be called, execute it
        if should_call_function and function_name == "get_weather":
            try:
                # Execute the weather function
                weather_result = get_weather(
                    function_args.get('latitude'),
                    function_args.get('longitude')
                )

                print(f"✅ Function executed. Result: {weather_result}")

                # Step 4: Send function response back to model
                follow_up_response = chat.send_message({
                    "function_response": {
                        "name": "get_weather",
                        "response": weather_result
                    }
                })

                # Step 5: Display final response
                print(f"🤖 Assistant: {follow_up_response.text}")

                return follow_up_response.text

            except Exception as func_error:
                print(f"❌ Error executing function: {func_error}")
                return f"Error getting weather data: {func_error}"

        else:
            # Model didn't call a function, just display the response
            print(f"🤖 Assistant: {response.text}")
            return response.text

    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        return f"Error: {e}"


# --------------------------------------------------------------
# Alternative: Simple approach without tools
# --------------------------------------------------------------

def simple_weather_query():
    """Alternative approach without function calling."""

    model = genai.GenerativeModel('gemini-1.5-flash')

    prompt = """
    You are a weather assistant. Based on general knowledge about Paris weather:
    - What's the typical weather like in Paris?
    - What temperature range is common?
    - Any seasonal considerations?

    Please provide a helpful weather overview for someone asking about Paris today.
    """

    response = model.generate_content(prompt)
    print(f"🌤️  Weather overview for Paris: {response.text}")
    return response.text


# --------------------------------------------------------------
# Run the examples
# --------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("WEATHER ASSISTANT DEMO")
    print("=" * 50)

    print("\n1. Testing with function calling:")
    result1 = run_weather_assistant()

    print("\n" + "=" * 30)
    print("\n2. Testing simple approach (no tools):")
    result2 = simple_weather_query()

    print("\n" + "=" * 30)
    print("\n3. Testing different coordinates:")
    # Test with different cities
    test_coordinates = [
        ("New York", 40.7128, -74.0060),
        ("Tokyo", 35.6762, 139.6503),
        ("Sydney", -33.8688, 151.2093)
    ]

    for city_name, lat, lon in test_coordinates:
        print(f"\n📍 Testing {city_name} ({lat}, {lon}):")
        weather_data = get_weather(lat, lon)
        print(f"   Simulated: {weather_data['temperature_2m']}°C, {weather_data['conditions']}")