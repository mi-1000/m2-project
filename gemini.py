import os
from dotenv import load_dotenv
from typing import Any, Generator, Literal, Union

from google import genai
from google.genai import types

load_dotenv()

class Gemini:
    def __init__(
        self,
        system_instructions: str = "You are a helpful assistant.",
        model: str = "gemini-2.5-flash",
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: Union[int, Literal["random"]] = "random",
        max_output_tokens: int = 65535,
        thinking_budget: int = 2048,
        contents: Union[list[types.Content], None] = None,
    ):
        self.system_instructions = system_instructions
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.max_output_tokens = max_output_tokens
        self.thinking_budget = thinking_budget
        self.contents: list[types.Content] = contents if contents is not None else []
        self.client = genai.Client(
            vertexai=True,
            project=os.environ.get("PROJECT", "missing-project-id"),
            location="global",
        )

    def add_turn(self, user_input: str) -> Generator[str, Any, None]:
        """
        Add a text user turn and stream back text chunks.
        """
        self.contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_input)],
            )
        )
        response = ""
        for chunk in self._generate_text(self.contents):
            response += chunk
            yield chunk
        # append full response to context
        self.contents.append(
            types.Content(
                role="model",
                parts=[types.Part.from_text(text=response)],
            )
        )

    def add_audio_turn(self, audio_bytes: bytes) -> Generator[str, Any, None]:
        """
        Add a user turn with raw audio and stream back the model's textual response.
        """
        self.contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")],
            )
        )

        response = ""
        for chunk in self._generate_text(self.contents):
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
        One-off text turn without context.
        """
        temp = [
            types.Content(role="user", parts=[types.Part.from_text(text=user_input)])
        ]
        for chunk in self._generate_text(temp):
            yield chunk

    def clear_context(self):
        """
        Reset conversation history.
        """
        self.contents.clear()

    def _generate_text(self, contents: list[types.Content]) -> Generator[str, Any, None]:
        cfg = types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed if isinstance(self.seed, int) else None,
            max_output_tokens=self.max_output_tokens,
            system_instruction=[types.Part.from_text(text=self.system_instructions)],
            thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget),
        )
        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=cfg,
        ):
            yield chunk.text
