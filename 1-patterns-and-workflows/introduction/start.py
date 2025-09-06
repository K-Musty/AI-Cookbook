import google.generativeai as genai

# Configure the library with your API key
genai.configure(api_key='') # <-- REPLACE THIS

# Create the model instance.
# We'll use 'gemini-1.5-flash' as it's fast, free, and great for getting started.
model = genai.GenerativeModel('gemini-1.5-flash')

# Start a chat session (it remembers context within this session)
chat = model.start_chat(history=[])

# Send your first message
response = chat.send_message("Hello! Can you explain what an API is in one short sentence?")

# Print the response
print("Gemini's Response:")
print(response.text)
print("\n") # Print a new line

# You can continue the conversation! The model remembers the context.
follow_up = chat.send_message("Now explain it like I'm 5 years old.")
print("Follow-up Response:")
