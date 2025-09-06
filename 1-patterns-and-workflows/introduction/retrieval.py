import json
import google.generativeai as genai


API_KEY = ""

# Configure Gemini
genai.configure(api_key=API_KEY)

# --------------------------------------------------------------
# WHAT IS RETRIEVAL? - Beginner Explanation
# --------------------------------------------------------------
"""
RETRIEVAL is like having a super-smart librarian:

1. You ask a question: "What's the return policy?"
2. The librarian (AI) knows to look in specific books (knowledge base)
3. The librarian finds the exact information you need
4. The librarian gives you a helpful answer using that information

Instead of making up answers, the AI retrieves facts from trusted sources!
"""


# --------------------------------------------------------------
# Define the knowledge base retrieval tool
# --------------------------------------------------------------

def search_kb(question: str):
    """Search our knowledge base for answers"""
    try:
        with open("kb.json", "r") as f:
            knowledge_base = json.load(f)

        # Simple search: look for matching questions
        for record in knowledge_base["records"]:
            if question.lower() in record["question"].lower():
                return record

        return {"answer": "I couldn't find information about that in our knowledge base."}

    except FileNotFoundError:
        return {"answer": "Knowledge base not available."}
    except Exception as e:
        return {"answer": f"Error accessing knowledge base: {str(e)}"}


# --------------------------------------------------------------
# Define the tool for Gemini
# --------------------------------------------------------------

kb_tool = {
    "function_declarations": [
        {
            "name": "search_kb",
            "description": "Search the knowledge base for answers to customer questions about store policies, shipping, and payments.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The customer's question to search for"
                    },
                },
                "required": ["question"]
            }
        }
    ]
}


# --------------------------------------------------------------
# SIMPLIFIED APPROACH: Manual Tool Handling (More Reliable)
# --------------------------------------------------------------

def ask_with_retrieval(question: str):
    """Ask a question with manual knowledge base retrieval"""

    print(f"\n👤 User asks: '{question}'")

    # Step 1: Check if this is a knowledge base question
    kb_keywords = ["return", "shipping", "payment", "policy", "accept", "ship"]
    is_kb_question = any(keyword in question.lower() for keyword in kb_keywords)

    if is_kb_question:
        print("🔍 Detected KB question - retrieving information...")

        # Step 2: Search knowledge base manually
        kb_result = search_kb(question)
        print(f"📚 KB found: {kb_result}")

        # Step 3: Use Gemini to format the answer nicely
        model = genai.GenerativeModel('gemini-1.5-flash')

        if "answer" in kb_result and "I couldn't find" not in kb_result["answer"]:
            # We found a good answer in KB
            prompt = f"""
            You are a helpful customer service assistant. Here is the official answer from our knowledge base:

            {kb_result['answer']}

            Please present this information in a friendly and helpful way to the customer who asked: "{question}"
            """
        else:
            # No good answer in KB, use general knowledge
            prompt = f"""
            You are a helpful customer service assistant. 
            A customer asked: "{question}"
            Our knowledge base doesn't have specific information about this.
            Please provide a helpful and honest response.
            """

        response = model.generate_content(prompt)
        print(f"🤖 Assistant: {response.text}")
        return response.text

    else:
        # General question - no retrieval needed
        print("🌍 General question - using AI knowledge")
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(question)
        print(f"🤖 Assistant: {response.text}")
        return response.text


# --------------------------------------------------------------
# DEMONSTRATION: Show Retrieval in Action
# --------------------------------------------------------------

def demonstrate_retrieval():
    """Show how retrieval works"""

    print("🔍 DEMONSTRATING KNOWLEDGE BASE RETRIEVAL")
    print("=" * 60)

    # Show KB contents
    print("\n📖 KNOWLEDGE BASE CONTENTS:")
    try:
        with open("kb.json", "r") as f:
            kb = json.load(f)
        for record in kb["records"]:
            print(f"  • {record['question']}")
    except:
        print("  (Could not load knowledge base)")

    print("\n" + "=" * 60)

    # Test questions
    test_questions = [
        # These should use knowledge base
        "What is the return policy?",
        "Do you ship internationally?",
        "What payment methods do you accept?",
        "How long do returns take?",

        # These should use general knowledge
        "What is the weather in Tokyo?",
        "Tell me a joke",
        "What's your favorite color?"
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. ", end="")
        ask_with_retrieval(question)
        if i < len(test_questions):
            print("-" * 40)


# --------------------------------------------------------------
# BONUS: Automatic Tool Version (Advanced)
# --------------------------------------------------------------

def ask_with_auto_tools(question: str):
    """Advanced version with automatic tool calling"""
    try:
        model = genai.GenerativeModel(
            'gemini-1.5-flash',
            tools=[kb_tool],
            system_instruction="You are a helpful assistant. Use search_kb for questions about returns, shipping, payments, or policies."
        )

        # Start chat WITHOUT automatic calling
        chat = model.start_chat(enable_automatic_function_calling=False)
        response = chat.send_message(question)

        # Check if model wants to call a function
        function_called = False
        for part in response.parts:
            if hasattr(part, 'function_call') and part.function_call:
                if part.function_call.name == "search_kb":
                    function_called = True
                    # Execute the function manually
                    kb_result = search_kb(part.function_call.args.get('question', ''))

                    # Send result back to model
                    follow_up = chat.send_message({
                        "function_response": {
                            "name": "search_kb",
                            "response": kb_result
                        }
                    })
                    return follow_up.text

        # If no function was called, return direct response
        if not function_called:
            return response.text

    except Exception as e:
        return f"Error: {str(e)}"


# --------------------------------------------------------------
# Run the demonstration
# --------------------------------------------------------------

if __name__ == "__main__":
    # Use the SIMPLE approach (most reliable)
    demonstrate_retrieval()

    print("\n" + "=" * 60)
    print("🚀 ADVANCED: Trying automatic tools...")
    print("=" * 60)

    # Try the advanced version
    try:
        advanced_response = ask_with_auto_tools("What is the return policy?")
        print(f"Advanced tool response: {advanced_response}")
    except Exception as e:
        print(f"Advanced tools failed (expected): {e}")