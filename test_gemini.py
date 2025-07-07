from gemini import GeminiChat

if __name__ == "__main__":
    chat = GeminiChat()
    print("Single turn test:")
    answer = chat.single_turn("What is the capital of France?")
    print("Q: What is the capital of France?")
    for chunk in answer:
        print("A:", chunk, end = "")

    print("\nMulti-turn test:")
    turns = [
        "Hello!",
        "Who do you think will be the next president of France?",
        "What did we just have a conversation about?"
    ]
    for question in turns:
        print(f"Q: {question}")
        print("A: ", end="")
        for chunk in chat.add_turn(question):
            print(chunk, end="", flush=True)
        print()
    chat.clear_context()
    print("\nContext cleared.")
    chat.system_instructions = "You should playfully answer by saying absolutely absurd things not even related to the question."
    absurd_question = "What did we just have a conversation about?"
    print(f"Q: {absurd_question}")
    print("A: ", end="")
    for chunk in chat.add_turn(absurd_question):
        print(chunk, end="", flush=True)
    print()
