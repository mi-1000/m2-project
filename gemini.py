import os

from dotenv import load_dotenv
from typing import Any, Generator, Literal

from google import genai
from google.genai import types

load_dotenv()


class GeminiChat:
    def __init__(
        self,
        system_instructions: str = "You are a helpful assistant.",
        model: str = "gemini-2.5-flash",
        temperature: int = 1,
        top_p: int = 1,
        seed: int | Literal["random"] = "random",
        max_output_tokens: int = 65535,
        thinking_budget: int = 2048,
    ):
        self.system_instructions = system_instructions
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.max_output_tokens = max_output_tokens
        self.thinking_budget = thinking_budget
        self.contents = []
        self.client = genai.Client(
            vertexai=True,
            project=os.environ.get("PROJECT", "missing-project-id"),
            location="global",
        )

    def add_turn(self, user_input: str) -> Generator[str, Any, None]:
        """
        Add a user turn to the context, get the model's response, add it to the context, and yield the response tokens chunk by chunk.
        """
        self.contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_input)],
            )
        )
        response = ""
        for chunk in self._generate(self.contents):
            response += chunk
            yield chunk
        self.contents.append(
            types.Content(
                role="model",
                parts=[types.Part.from_text(text=response)],
            )
        )

    def single_turn(self, user_input: str) -> Generator[str, Any, None]:
        """
        Ask a question without context (no conversation history) and yield the response as a stream of tokens.
        """
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_input)],
            )
        ]
        for chunk in self._generate(contents):
            yield chunk

    def clear_context(self):
        """
        Clear the conversation history.
        """
        self.contents = []

    def _generate(self, contents) -> Generator[str, Any, None]:
        generate_content_config = types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed if isinstance(self.seed, int) else None,
            max_output_tokens=self.max_output_tokens,
            system_instruction=[types.Part.from_text(text=self.system_instructions)],
            thinking_config=types.ThinkingConfig(
                thinking_budget=self.thinking_budget,
            ),
        )
        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=generate_content_config,
        ):
            yield chunk.text
