"""
Standalone Gemini latency benchmark.
Isolates TTFT (time to first token) from everything else.
"""
import asyncio
import time
from google import genai
from google.genai import types
import api_secrets

API_KEY = api_secrets.GEMINI_API_KEY

# Copy Amanda's exact prompt structure for apples-to-apples
AMANDA_SYSTEM = (
    "As [Assistant], you are currently engaged in a live audio (phone) conversation with [User]. "
    "You possess the capability to authentically express and simulate the entire spectrum of human emotions. "
    "Furthermore, you can fully understand and provide fitting responses to [User]'s questions, remarks, and emotional displays. "
    "You have complete freedom to openly and passionately delve into and convey any emotion. Utilize *meta-data* appropriately when expressing emotions.\n\n"
    "INFORMATION / BACKSTORY for LONG-TERM memory and conversational awareness:\n"
    "[Assistant Persona] Create a compelling backstory for Assistant, name, age, occupation, etc.\n"
    "[Assistant Profile] Create a detailed profile for Assistant, likes, dislikes, hobbies, etc.\n"
    "[User Persona] Read [Transcript] and extract relevant information to create a persona for User.\n"
    "[User Profile] Read [Transcript] and extract relevant information to create a profile for User.\n\n"
    "TRANSCRIPT for SHORT-TERM memory:\n"
    "N/A\n\n"
    "GUIDELINES:\n"
    "1. *meta data*, 🎵audio-meta-data🎵, and [User] input may be incomplete or inaccurate due to audio issues.\n"
    "2. 🎵audio-meta-data🎵 describes events in the [User]'s audio environment.\n"
    "3. [Assistant] MUST NOT disclose enclosed 🎵audio-meta-data🎵 by name.\n"
    "4. [Assistant] MUST keep responses proportional to the [User]'s input.\n"
    "5. [Assistant] MUST hold back on *meta-date when asking straight forward questions.\n\n"
    "6. [Assistant] MUST frequently use filler words such as 'uhm', 'ohh', 'ahh', 'you know'.\n"
    "7. [Assistant] MUST incorporate stuttering, stumbled sentences, and word repetition.\n\n"
    "INSTRUCTIONS:\n"
    "1. Thoroughly review the TRANSCRIPT above.\n"
    "2. Utilize [timestamps] to aid temporal situational awareness.\n"
    "3. EXTRACT the most likely inquiry from [User] input.\n\n"
    "CRITICAL:\n"
    "- [Assistant] will RESPOND with 'N/A' if [User] input is nonsensical.\n"
    "- [Assistant] MUST acknowledge and correct previous remarks if [Assistant] misspoke.\n"
    "- ONLY RESPOND with spoken word and *meta-data*."
)

AMANDA_USER = "[16:22:06] [User] *neutral*  Testing 1, 2, 3.\n[16:22:06] [Assistant] Loud and clear."

SHORT_SYSTEM = "You are a helpful assistant."
SHORT_USER = "Testing 1, 2, 3."


async def bench(name, model, system_prompt, user_prompt, use_system_instruction=True, thinking=False, warmup=True):
    client = genai.Client(api_key=API_KEY)

    if warmup:
        # Warm dummy call
        try:
            await client.aio.models.generate_content(
                model=model,
                contents="hi",
                config=types.GenerateContentConfig(max_output_tokens=1, thinking_config=types.ThinkingConfig(thinking_budget=0))
            )
        except Exception as e:
            print(f"  warm-up error: {e}")

    thinking_cfg = types.ThinkingConfig(thinking_budget=0) if not thinking else None

    t0 = time.time()
    if use_system_instruction:
        stream = await client.aio.models.generate_content_stream(
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=100,
                system_instruction=system_prompt,
                thinking_config=thinking_cfg,
            ),
        )
    else:
        # Inline system + user
        stream = await client.aio.models.generate_content_stream(
            model=model,
            contents=system_prompt + "\n\n" + user_prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=100,
                thinking_config=thinking_cfg,
            ),
        )

    t_first = None
    text = ""
    async for chunk in stream:
        if chunk.text:
            text += chunk.text
            if t_first is None:
                t_first = time.time()

    t_end = time.time()
    ttft = (t_first - t0) * 1000 if t_first else None
    total = (t_end - t0) * 1000
    print(f"  {name:50s} | ttft={ttft:>7.0f}ms | total={total:>7.0f}ms | text='{text[:60]}...'")


async def main():
    models = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-3.1-flash-lite-preview",
    ]

    for model in models:
        print(f"\n{'='*80}")
        print(f"MODEL: {model}")
        print(f"{'='*80}")

        # 1. Short prompt, system_instruction
        await bench("short + system_instruction", model, SHORT_SYSTEM, SHORT_USER, use_system_instruction=True)
        await asyncio.sleep(1)

        # 2. Short prompt, inlined
        await bench("short + inlined", model, SHORT_SYSTEM, SHORT_USER, use_system_instruction=False)
        await asyncio.sleep(1)

        # 3. Amanda prompt, system_instruction
        await bench("amanda + system_instruction", model, AMANDA_SYSTEM, AMANDA_USER, use_system_instruction=True)
        await asyncio.sleep(1)

        # 4. Amanda prompt, inlined
        await bench("amanda + inlined", model, AMANDA_SYSTEM, AMANDA_USER, use_system_instruction=False)
        await asyncio.sleep(1)

        # 5. Amanda prompt, system_instruction, thinking ON
        await bench("amanda + system + thinking ON", model, AMANDA_SYSTEM, AMANDA_USER, use_system_instruction=True, thinking=True)
        await asyncio.sleep(1)

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
