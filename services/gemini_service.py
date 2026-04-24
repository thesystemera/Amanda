import time
from google import genai
from google.genai import types
import config

class GeminiService:

    def __init__(self):
        config.custom_print("Lifespan", "GeminiService: initializing client...")
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        self._warm = False

    async def warm_up(self):
        if self._warm:
            return
        config.custom_print("Lifespan", "GeminiService: warming up model pools...")
        t0 = time.time()

        for name, model in (("task", config.GEMINI_TASK_MODEL_NAME),
                            ("chat", config.GEMINI_CHAT_MODEL_NAME)):
            try:
                await self.client.aio.models.generate_content(
                    model=model,
                    contents="1",
                    config=types.GenerateContentConfig(max_output_tokens=1)
                )
                config.custom_print("Lifespan", f"GeminiService: {name} pool ready ({model}).")
            except Exception as e:
                config.custom_print("Error", f"GeminiService: {name} warm-up failed: {e}")

        self._warm = True
        config.custom_print("Lifespan", f"GeminiService: ready ({(time.time() - t0) * 1000:.0f}ms).")

    async def generate_content(self, model, contents, config):
        response = await self.client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config
        )
        return response.text.strip() if response.text else None

    async def generate_content_stream(self, model, contents, config):
        stream = await self.client.aio.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config
        )
        async for chunk in stream:
            if chunk.text:
                yield chunk.text

    async def count_tokens(self, model, contents):
        try:
            tok = await self.client.aio.models.count_tokens(
                model=model,
                contents=contents
            )
            return tok.total_tokens
        except Exception:
            return None