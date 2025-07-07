import os

from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types

def generate(
    prompt: str,
    system_instructions: str,
    model: str = "gemini-2.5-flash",
    temperature: int = 1,
    top_p: int = 1,
    seed: int = 0,
    max_output_tokens: int = 65535,
    thinking_budget: int = 2048,
    ):
    client = genai.Client(
        vertexai=True,
        project=os.environ.get("PROJECT", "missing-project-id"),
        location="global",
    )

    model = model
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        )
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        max_output_tokens=max_output_tokens,
        system_instruction=[types.Part.from_text(text=system_instructions)],
        thinking_config=types.ThinkingConfig(
            thinking_budget=thinking_budget,  # Max tokens used for thinking
        ),
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        yield chunk.text


if __name__ == "__main__":
    for token in generate("Tell me briefly about the history of France.", "You are an expert in European history."):
        print(token, end="", flush=True)
