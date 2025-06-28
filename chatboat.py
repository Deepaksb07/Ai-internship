
def chatbot_response(user_input):
    user_input = user_input.lower()

    if "hello" in user_input:
        return "Hello! How can I help you today?"
    if "hi" in user_input:
        return "Hi! How can I help you today?"
    elif "what is todays whether" in user_input:
        return "It's sunny and 28°C."
    elif"thankyou" in user_input:
        return "your welcome!"
    elif "bye" in user_input:
        return "Goodbye! Have a great day!"
    else:
        return "Sorry, I didn't understand that. Can you try something else?"
    
    # Get user input
def run_chatbot():
    print("Welcome to SimpleBot! (type 'bye' to exit)")

    while True:
        user_input = input("You: ")
        response = chatbot_response(user_input)
        print("Bot:", response)

        if "bye" in user_input.lower():
            break
        
#  Run and test
if __name__ == "__main__":
    run_chatbot()

