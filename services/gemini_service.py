import time
from google import genai
from google.genai import types
import config


class GeminiService:

    def __init__(self):
        config.custom_print("Lifespan", "GeminiService: initializing client...")
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        self._warm = False
        self._caches = {}

    def create_cache(self, name: str, system_instruction: str, model: str = None):
        """Create a cached context for a static system prompt."""
        model = model or config.GEMINI_CHAT_MODEL_NAME
        try:
            cache = self.client.caches.create(
                model=model,
                config=types.CreateCachedContentConfig(
                    system_instruction=system_instruction,
                    display_name=f"amanda_{name}",
                    ttl="3600s",
                )
            )
            self._caches[name] = cache
            config.custom_print("Lifespan", f"GeminiService: cache '{name}' created ({model}).")
            return cache
        except Exception as e:
            config.custom_print("Error", f"GeminiService: cache '{name}' failed: {e}")
            return None

    def refresh_cache(self, name: str, system_instruction: str, model: str = None):
        """Delete and recreate a cache (used when cached content changes, e.g. persona update)."""
        old = self._caches.pop(name, None)
        if old:
            try:
                self.client.caches.delete(name=old.name)
            except Exception:
                pass
        return self.create_cache(name, system_instruction, model)

    def cleanup_caches(self):
        """Delete all active caches to stop storage billing. Call on shutdown."""
        for name, cache in list(self._caches.items()):
            try:
                self.client.caches.delete(name=cache.name)
                config.custom_print("Lifespan", f"GeminiService: cache '{name}' deleted.")
            except Exception as e:
                config.custom_print("Error", f"GeminiService: failed to delete cache '{name}': {e}")
        self._caches.clear()

    def get_cache_name(self, name: str) -> str | None:
        cache = self._caches.get(name)
        return cache.name if cache else None

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
