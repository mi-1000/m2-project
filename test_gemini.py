from typing import Literal, Union
from gemini import Gemini


def test_fun():
    chat = Gemini()
    print("Single turn test:")
    answer = chat.single_turn("What is the capital of France?")
    print("Q: What is the capital of France?")
    for chunk in answer:
        print("A:", chunk, end="")

    print("\nMulti-turn test:")
    turns = [
        "Hello!",
        "Who do you think will be the next president of France?",
        "What did we just have a conversation about?",
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


SYSTEM_PROMPT = """
You are a friendly and knowledgeable language tutor. Help the user practice {language} by having a short, natural conversation about “{topic}” at level {difficulty_level}.

Guidelines:
1. Speak mostly in {language}, using simple words and grammar for level {difficulty_level}.
2. Ask short, open questions about {topic}. Encourage the user to answer in their own words.
3. If the user makes a mistake, gently correct them:
   - Give the correct sentence.
   - Briefly explain the rule (in {explanation_language}).
   - Optionally, give a quick tip or example.

If asked, give cultural info, vocabulary, or grammar tips. Always be encouraging and keep answers short and clear.

Start by greeting the user in {language} and asking a simple question about “{topic}” at level {difficulty_level}."""


def launch_dialogue(
    conversation_language: str,
    topic: str,
    difficulty_level: Literal["A1", "A2", "B1", "B2", "C1", "C2"],
    explanation_language: Union[str, None] = None,
):
    chat = Gemini(
        system_instructions=SYSTEM_PROMPT.format(
            language=conversation_language,
            topic=topic,
            difficulty_level=difficulty_level,
            explanation_language=explanation_language or conversation_language,
        )
    )
    print("Dialogue launched with system instructions:", chat.system_instructions)
    return chat


if __name__ == "__main__":
    chat = launch_dialogue("Magyar", "Pékségbe menni", "A2", "Français")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting dialogue.")
            break
        print("Chatbot: ", end="")
        for chunk in chat.add_turn(user_input):
            print(chunk, end="", flush=True)
        print()
