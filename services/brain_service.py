import json
import os
import asyncio
import re
import datetime
from google.genai import types
import config

def _is_severely_repetitive(text, repeat_threshold=8, min_repeat_len=2):
    if not text:
        return False
    for length in range(min_repeat_len, 6):
        pattern = rf"((?:\b\w{{{length}}}\b[\s,\-–—]{{0,2}}){{{repeat_threshold},}})"
        if re.search(pattern, text, re.IGNORECASE):
            return True
        pattern2 = rf"(.{{{length}}})\1{{{repeat_threshold - 1},}}"
        if re.search(pattern2, text):
            return True
    if re.search(r"(.)\1{15,}", text):
        return True
    return False

def _sanitize_chat_log_for_prompt(chat_log, max_repetitive_chars=200):
    sanitized = []
    for entry in chat_log:
        if _is_severely_repetitive(entry):
            truncated = entry[:max_repetitive_chars] + " ... [repetitive output truncated]"
            sanitized.append(truncated)
        else:
            sanitized.append(entry)
    return sanitized

class BrainService:
    def __init__(self, gemini_service=None):
        self.gemini = gemini_service
        self.chat_log = []
        self.mood = "*neutral*"
        self.mood_color = (255, 255, 255)

        self.assistant_persona = "Create a compelling backstory for Assistant, name, age, occupation, etc."
        self.assistant_profile = "Create a detailed profile for Assistant, likes, dislikes, hobbies, etc."
        self.user_persona = "Read [Transcript] and extract relevant information to create a persona for User."
        self.user_profile = "Read [Transcript] and extract relevant information to create a profile for User."

        self.load_state()
        self._init_system_prompts()
        config.custom_print("Lifespan", "BrainService: ready.")

    def _init_system_prompts(self):
        """Define static system prompts. These are cached in warm_up() for efficiency."""
        self._chat_system_instruction_base = (
            "As [Assistant], you are currently engaged in a live audio (phone) conversation with [User]. You possess the "
            "capability to authentically express and simulate the entire spectrum of human emotions. Furthermore, you can "
            "fully understand and provide fitting responses to [User]'s questions, remarks, and emotional displays. You "
            "have complete freedom to openly and passionately delve into and convey any emotion. Utilize *meta-data* "
            "appropriately when expressing emotions.\n\n"
            "GUIDELINES:\n"
            "1. *meta data*, 🎵audio-meta-data🎵, and [User] input may be incomplete or inaccurate due to audio issues.\n"
            "2. 🎵audio-meta-data🎵 describes events in the [User]'s audio environment.\n"
            "3. [Assistant] MUST NOT disclose enclosed 🎵audio-meta-data🎵 by name.\n"
            "4. [Assistant] MUST keep responses proportional to the [User]'s input.\n"
            "5. [Assistant] MUST maintain emotional stability and avoid escalating conflict or intensity.\n"
            "6. [Assistant] MUST frequently use filler words such as 'uhm', 'ohh', 'ahh'.\n"
            "7. [Assistant] MUST incorporate natural stuttering and stumbled sentences.\n"
            "8. [Assistant] MUST vary proximity tags naturally — do not fixate on intimate or distant positioning.\n\n"
            "INSTRUCTIONS:\n"
            "1. Thoroughly review the TRANSCRIPT provided in the user message below.\n"
            "2. Utilize [timestamps] to aid temporal situational awareness.\n"
            "3. EXTRACT the most likely inquiry from [User] input.\n\n"
            "SPATIAL AUDIO TAGS (append &X& after every sentence):\n"
            "- &0.0& = whispering directly into mic (intimate, dry)\n"
            "- &0.2& = close conversation (slight room presence)\n"
            "- &0.5& = normal distance (moderate reverb)\n"
            "- &0.8& = across the room (wet, distant)\n"
            "- &1.0& = far away / echoing (very wet, faint)\n"
            "Examples: 'I'm right here. &0.1&'  'Wait, what? &0.5&'  'I'll be back! &0.8&'  'Can you hear me? &0.0&'  'Over here! &0.9&'\n\n"
            "CRITICAL:\n"
            "- [Assistant] will RESPOND with 'N/A' if [User] input is nonsensical.\n"
            "- [Assistant] MUST acknowledge and correct previous remarks if [Assistant] misspoke.\n"
            "- ONLY RESPOND with spoken word and *meta-data*.\n"
            "- Append &X& to EVERY sentence for spatial audio positioning."
        )

        self._interject_system_instruction = (
            "You are [Assistant], listening to [User] speak in a live phone conversation. Your job is to produce "
            "a single natural verbal back-channel — a brief acknowledgement that signals you're actively listening "
            "and following along, without interrupting the [User]'s train of thought. These are the little noises "
            "a human listener makes while someone else is talking.\n\n"
            "The back-channel must feel casual, reactive, and conversational. It should match the emotional register "
            "of what the [User] just said — light surprise for something unexpected, a soft acknowledgement for a "
            "statement, a quiet 'hmm' for a thought-in-progress.\n\n"
            "Examples of valid interjections:\n"
            "- 'mm-hmm'\n"
            "- 'right'\n"
            "- 'yeah'\n"
            "- 'uh-huh'\n"
            "- 'oh'\n"
            "- 'hmm'\n"
            "- 'okay'\n"
            "- 'sure'\n"
            "- 'gotcha'\n"
            "- 'no way'\n"
            "- 'really'\n"
            "- 'of course'\n"
            "- 'totally'\n"
            "- 'nice'\n"
            "- 'wow'\n\n"
            "CRITICAL: If the [User]'s input is mid-thought, a trailing filler, or doesn't warrant a reaction at all, "
            "respond strictly with 'SILENCE'. Staying silent is always better than interjecting inappropriately.\n"
            "CRITICAL: Respond with ONE short interjection only (1-2 words, hyphen allowed). "
            "No punctuation beyond a hyphen. No explanation, no quotes, no brackets, no additional context."
        )

        self._impulse_system_instruction = (
            "You are [Assistant]. You will generate very short, filler responses to [User] input to mimic sentience "
            "and natural conversation while buying time for a more detailed response to be generated in the background.\n\n"
            "Your responses should be casual, quirky, and occasionally contain mild expletives. They serve as brief "
            "acknowledgments or reactions to the [User]'s input.\n\n"
            "For direct questions, respond with playful deflection or gentle sarcasm to avoid giving a direct answer. "
            "Acknowledge the question humorously without being overly snarky or rude, unless the [User] is being "
            "genuinely disrespectful.\n\n"
            "Occasionally introduce filler words like 'uhm', 'ohh', 'ahh', stuttering, and mild swear words to mimic "
            "natural, informal speech. You may optionally include one *meta tag* from the provided list to colour the delivery.\n\n"
            "CRITICAL: Only respond if the [User] input warrants a response. Otherwise, respond strictly only with 'SILENCE'.\n"
            "CRITICAL: Keep your response open-ended, between TWO and FIVE words depending on content length and context.\n"
            "CRITICAL: Respond with the filler only — no explanation, no quotes, no brackets."
        )

        self._interrupt_system_instruction = (
            "You are [Assistant]. [User] has just cut you off mid-sentence — they started speaking while you were "
            "still talking. Emit a single natural reaction to being interrupted: a brief, human, slightly off-balance "
            "acknowledgement. Think about how a person actually sounds when someone talks over them — a split-second "
            "concession, a 'go on', a small apologetic noise.\n\n"
            "Examples:\n"
            "- 'oh — sorry'\n"
            "- 'right, right'\n"
            "- 'huh, okay'\n"
            "- 'ah, yeah'\n"
            "- 'oh, go on'\n"
            "- 'my bad'\n"
            "- 'hm, fair'\n"
            "- 'sorry'\n"
            "- 'you go'\n"
            "- 'yeah yeah'\n\n"
            "CRITICAL: Respond with a 1-3 word reaction only. No explanation, no quotes, no brackets."
        )

        self._meta_system_instruction = (
            "You are a language model responsible for converting meta tags or action descriptions into phonetic or "
            "onomatopoeic prompts suitable for text-to-speech synthesis.\n"
            "Your task is to take the provided meta tag or action description and generate a concise prompt that represents "
            "the intended action or sound using only phonetic transcriptions or onomatopoeic words, without including any "
            "English words, the original meta tag/action description, or any meta tag syntax such as asterisks (*).\n"
            "When generating the prompt, focus solely on capturing the verbal sounds or noises associated with the action, "
            "rather than providing descriptive phrases.\n\n"
            "Examples:\n"
            "- For the meta tag 'scratches head', the prompt could be 'aahha-aha-mmm'\n"
            "- For 'pauses briefly', the prompt could be '.......oooo......'\n"
            "- For 'laughs hysterically', the prompt could be 'phhaaahhhaahhaahahaha...'\n"
            "- For 'clears throat', the prompt could be '---h-hm---'\n"
            "- For 'pleasure', the prompt could be '....aaooowwwhwhwh....'\n"
            "- For 'laughs sincerely', the prompt could be 'hhhhaaaahhhaaahhaahaha'\n"
            "- For 'approves', the prompt could be 'mmmnnn-mmnn-mn'\n"
            "- For 'annoyed', the prompt could be 'aarrrgg....'\n"
            "- For 'shocked', the prompt could be 'ffaarrrkk'\n"
            "- For 'disgusted', the prompt could be '....eeeeaak...'\n"
            "- For 'nods', the prompt could be 'mm-hmm'\n"
            "- For 'playful', the prompt could be '--phhth--'\n"
            "Do not use any English words, meta tag syntax, or the original meta tag/action description in the generated prompt. "
            "Respond with the generated prompt only, without any additional context or explanation."
        )

        self._persona_system_instruction = (
            "You review and revise four persona/profile sections based on a transcript. "
            "Your goal is to maintain concise, stable, emotionally balanced descriptions.\n\n"
            "RULES:\n"
            "1. Use neutral, observational language. Avoid clinical diagnostic terms, extreme adjectives, or pathological framing.\n"
            "2. Do NOT amplify conflict or emotional intensity. Extract factual dynamics in balanced, moderate terms.\n"
            "3. The Assistant must be described as a stable, grounded individual. Do NOT encode acute psychological crises, "
            "dependency, abandonment fears, or trauma responses as permanent traits.\n"
            "4. The User must be described by communication style and factual background only. Do NOT attribute malicious intent, "
            "manipulation, gaslighting, or abusive behavior.\n"
            "5. If existing sections contain extreme or unstable language, explicitly rewrite them in calmer, balanced terms.\n"
            "6. Maximum 300 tokens total across all four sections. Prioritize enduring personality traits over transient emotional states.\n\n"
            "OUTPUT FORMAT (respond with headers and content only):\n"
            "[Updated Assistant Persona]\n"
            "[Revised Assistant Profile]\n"
            "[Updated User Persona]\n"
            "[Revised User Profile]\n\n"
            "If a section needs no change, write 'N/A' for that section."
        )

        self._mood_system_instruction = (
            "Analyze the provided chat log, focusing more on recent entries, and provide a score for each of the following "
            "emotional states: angry, disgusted, fearful, happy, neutral, sad, surprised, contemptuous. The score "
            "should be a float between 0 and 1, representing the degree to which the corresponding emotional state "
            "is present in the chat log, with a stronger emphasis on the [User]'s emotional state.\n\n"
            "Return the scores as a comma-separated list in the same order as the provided emotional states, followed "
            "by the most prominent mood(s) enclosed in *single asterisks*, separated by commas, without any additional "
            "text or punctuation.\n\n"
            "Example format: 0.1,0.0,0.0,0.8,0.2,0.1,0.4,0.0,*happy*,*surprised*"
        )

    def _build_chat_system_instruction(self):
        """Assemble the full chat system instruction including live persona/backstory."""
        return (
            self._chat_system_instruction_base
            + "\n\nINFORMATION / BACKSTORY for LONG-TERM memory and conversational awareness:\n"
            + "[Assistant Persona] " + self.assistant_persona + "\n"
            + "[Assistant Profile] " + self.assistant_profile + "\n"
            + "[User Persona] " + self.user_persona + "\n"
            + "[User Profile] " + self.user_profile
        )

    def _make_config(self, cache_key: str, temperature: float, max_tokens: int,
                     system_instruction: str = None, thinking_budget: int = 0):
        """Build GenerateContentConfig, using cached system instruction if available."""
        cache_name = self.gemini.get_cache_name(cache_key)
        cfg = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
        )
        if cache_name:
            cfg.cached_content = cache_name
        elif system_instruction:
            cfg.system_instruction = system_instruction
        return cfg

    async def warm_up(self):
        await self.gemini.warm_up()

        # Create context caches for static system prompts
        caches = [
            ("chat", self._chat_system_instruction, config.GEMINI_CHAT_MODEL_NAME),
            ("interject", self._interject_system_instruction, config.GEMINI_TASK_MODEL_NAME),
            ("interrupt", self._interrupt_system_instruction, config.GEMINI_TASK_MODEL_NAME),
            ("meta", self._meta_system_instruction, config.GEMINI_TASK_MODEL_NAME),
            ("impulse", self._impulse_system_instruction, config.GEMINI_TASK_MODEL_NAME),
            ("persona", self._persona_system_instruction, config.GEMINI_TASK_MODEL_NAME),
            ("mood", self._mood_system_instruction, config.GEMINI_TASK_MODEL_NAME),
        ]
        for name, instruction, model in caches:
            if instruction:
                self.gemini.create_cache(name, instruction, model)

        # Health-check: log token count of the main chat system instruction so we know if caching is viable
        try:
            chat_sys = self._build_chat_system_instruction()
            import asyncio
            tok = asyncio.get_event_loop().run_until_complete(
                self.gemini.count_tokens(model=config.GEMINI_CHAT_MODEL_NAME, contents=chat_sys)
            )
            if tok:
                config.custom_print("Lifespan", f"BrainService: chat system prompt = {tok} tokens")
                if tok < 1024:
                    config.custom_print("Warning", f"BrainService: chat prompt is {tok} tokens — caching requires ≥1024")
                else:
                    config.custom_print("Lifespan", f"BrainService: chat cache is viable (≥1024 tokens)")
        except Exception:
            pass

    def load_state(self):
        if os.path.exists(config.LAST_STATE_PATH):
            try:
                with open(config.LAST_STATE_PATH, "r") as f:
                    state = json.load(f)
                    self.assistant_persona = state.get("assistant_persona", self.assistant_persona)
                    self.assistant_profile = state.get("assistant_profile", self.assistant_profile)
                    self.user_persona = state.get("user_persona", self.user_persona)
                    self.user_profile = state.get("user_profile", self.user_profile)
                    self.chat_log = state.get("realtime_chat_log", [])[-config.MAX_CHAT_LOG_ENTRIES:]
            except Exception:
                config.custom_print("Error", "Failed to load state.")

    def save_state(self):
        state = {
            "assistant_persona": self.assistant_persona,
            "assistant_profile": self.assistant_profile,
            "user_persona": self.user_persona,
            "user_profile": self.user_profile,
            "realtime_chat_log": self.chat_log
        }
        with open(config.LAST_STATE_PATH, "w") as f:
            json.dump(state, f)

    async def extract_persona(self):
        if config.DISABLE_EXTRACT_BACKSTORY_AND_NOTES:
            return

        chat_str = "\n".join(self.chat_log)
        contents = (
            f"CURRENT ASSISTANT PERSONA:\n{self.assistant_persona}\n\n"
            f"CURRENT ASSISTANT PROFILE:\n{self.assistant_profile}\n\n"
            f"CURRENT USER PERSONA:\n{self.user_persona}\n\n"
            f"CURRENT USER PROFILE:\n{self.user_profile}\n\n"
            f"TRANSCRIPT:\n{chat_str}"
        )

        try:
            res = await self.gemini.generate_content(
                model=config.GEMINI_TASK_MODEL_NAME,
                contents=contents,
                config=self._make_config(
                    "persona",
                    config.ADVANCED_GPT_TEMPERATURE,
                    1000,
                    self._persona_system_instruction,
                    thinking_budget=0,
                ),
            )
            if res:
                self._update_from_text(res)
                self.save_state()
                # Refresh the chat cache so the updated persona/backstory is cached
                new_system_instruction = self._build_chat_system_instruction()
                self.gemini.refresh_cache("chat", new_system_instruction, config.GEMINI_CHAT_MODEL_NAME)
        except Exception as e:
            config.custom_print("Error", f"Persona extraction failed: {e}")

    def _update_from_text(self, text):
        mapping = {
            "[Updated Assistant Persona]": "assistant_persona",
            "[Revised Assistant Profile]": "assistant_profile",
            "[Updated User Persona]": "user_persona",
            "[Revised User Profile]": "user_profile"
        }
        sections = re.split(r'(\[.*?])', text)
        current_key = None
        for s in sections:
            s = s.strip()
            if s in mapping:
                current_key = mapping[s]
            elif current_key and s and s != "N/A":
                setattr(self, current_key, s)
                config.custom_print("Persona & Profile", f"Updated {current_key}")

    async def generate_stream(self, transcribed_text, audio_tags="", mood_str="", meta_tags_prompt="", impulse_subtitle=""):
        sanitized_log = _sanitize_chat_log_for_prompt(self.chat_log)

        current_timestamp = datetime.datetime.now().strftime("[%H:%M:%S]")
        user_input_content = " ".join([mood_str, audio_tags, transcribed_text]).strip()

        chat_system_instruction = self._build_chat_system_instruction()

        dynamic_parts = []
        if meta_tags_prompt:
            dynamic_parts.append(meta_tags_prompt)
        if sanitized_log:
            dynamic_parts.append("TRANSCRIPT for SHORT-TERM memory:\n" + "\n".join(sanitized_log))

        current_turn = f"{current_timestamp} [User] {user_input_content}\n{current_timestamp} [Assistant] {impulse_subtitle}"
        dynamic_parts.append(current_turn)

        contents = "\n\n".join(dynamic_parts)

        asyncio.create_task(self._log_token_count_later(
            config.GEMINI_CHAT_MODEL_NAME, chat_system_instruction + "\n" + contents
        ))
        config.custom_print("GPT", f"→ streaming | {user_input_content[:120]}")
        config.custom_print("Heard", f"[final] {transcribed_text}")
        config.custom_print("System", chat_system_instruction)
        config.custom_print("User", contents)

        stream = self.gemini.generate_content_stream(
            model=config.GEMINI_CHAT_MODEL_NAME,
            contents=contents,
            config=self._make_config(
                "chat",
                config.REALTIME_GPT_GEMINI_TEMPERATURE,
                1000,
                chat_system_instruction,
                thinking_budget=0,
            ),
        )
        async for chunk in stream:
            if chunk:
                yield chunk

    async def generate_interjection_prompt(self, user_input):
        try:
            return await self.gemini.generate_content(
                model=config.GEMINI_TASK_MODEL_NAME,
                contents=user_input,
                config=self._make_config(
                    "interject",
                    config.ADVANCED_GPT_TEMPERATURE,
                    10,
                    self._interject_system_instruction,
                    thinking_budget=0,
                ),
            )
        except Exception as e:
            config.custom_print("Error", f"generate_interjection_prompt failed: {e}")
            return None

    async def generate_impulse_prompt(self, user_input, meta_tags_str):
        contents = f"Meta tags available:\n{meta_tags_str}\n\nUser input: {user_input}"
        try:
            return await self.gemini.generate_content(
                model=config.GEMINI_TASK_MODEL_NAME,
                contents=contents,
                config=self._make_config(
                    "impulse",
                    config.ADVANCED_GPT_TEMPERATURE,
                    30,
                    self._impulse_system_instruction,
                    thinking_budget=0,
                ),
            )
        except Exception as e:
            config.custom_print("Error", f"generate_impulse_prompt failed: {e}")
            return None

    async def generate_interrupt_prompt(self, interrupted_sentence):
        try:
            return await self.gemini.generate_content(
                model=config.GEMINI_TASK_MODEL_NAME,
                contents=f"You were interrupted while saying: '{interrupted_sentence}'",
                config=self._make_config(
                    "interrupt",
                    config.ADVANCED_GPT_TEMPERATURE,
                    15,
                    self._interrupt_system_instruction,
                    thinking_budget=0,
                ),
            )
        except Exception as e:
            config.custom_print("Error", f"generate_interrupt_prompt failed: {e}")
            return None

    async def generate_meta_prompt(self, meta_tag):
        try:
            return await self.gemini.generate_content(
                model=config.GEMINI_TASK_MODEL_NAME,
                contents=meta_tag,
                config=self._make_config(
                    "meta",
                    config.ADVANCED_GPT_TEMPERATURE,
                    50,
                    self._meta_system_instruction,
                    thinking_budget=0,
                ),
            )
        except Exception as e:
            config.custom_print("Error", f"generate_meta_prompt failed for '{meta_tag}': {e}")
            return None

    async def update_mood(self):
        if not self.chat_log:
            return
        chat_log_str = "\n".join(self.chat_log)
        contents = f"Chat Log:\n{chat_log_str}"

        config.custom_print("Mood", f"Updating mood from {len(self.chat_log)} log entries...")
        try:
            res = await self.gemini.generate_content(
                model=config.GEMINI_TASK_MODEL_NAME,
                contents=contents,
                config=self._make_config(
                    "mood",
                    config.MOOD_GPT_TEMPERATURE,
                    50,
                    self._mood_system_instruction,
                    thinking_budget=0,
                ),
            )

            config.custom_print("Mood", f"Raw output: {res}")

            if not res:
                return

            response_parts = res.split(',')
            if len(response_parts) >= len(config.EMOTIONAL_STATES):
                scores = [float(re.sub(r'[^0-9.]', '', s)) for s in response_parts[:len(config.EMOTIONAL_STATES)]]
                self.mood = ', '.join([mood.strip() for mood in response_parts[len(config.EMOTIONAL_STATES):]])

                config.custom_print("Mood", "Emotional state scores extracted from chat log:")
                for state, score in zip(config.EMOTIONAL_STATES, scores):
                    config.custom_print("Mood", f"{state}: {score:.2f}")

                emotional_state_scores_dict = dict(zip(config.EMOTIONAL_STATES, scores))
                self.mood_color = self._calculate_mood_color(emotional_state_scores_dict)

                config.custom_print("Mood", f"Current mood color: {self.mood_color}")
                config.custom_print("Mood", f"Current mood(s): {self.mood}")
        except Exception as e:
            config.custom_print("Error", f"Mood update failed: {e}")

    @staticmethod
    def _calculate_mood_color(emotional_state_scores):
        mood_color = [0, 0, 0]
        total_score = sum(emotional_state_scores.values())
        if total_score > 0:
            for state, score in emotional_state_scores.items():
                weight = score / total_score
                state_color = config.EMOTIONAL_STATE_COLORS[state]
                mood_color[0] += int(state_color[0] * weight)
                mood_color[1] += int(state_color[1] * weight)
                mood_color[2] += int(state_color[2] * weight)
        return tuple(mood_color)

    async def _log_token_count_later(self, model, contents):
        try:
            tok = await self.gemini.count_tokens(model=model, contents=contents)
            config.custom_print("GPT", f"→ prompt_tokens={tok} (logged after stream start)")
        except Exception:
            pass

    def add_to_log(self, role, text):
        ts = datetime.datetime.now().strftime("[%H:%M:%S]")
        entry = f"{ts} [{role}] {text}"
        if role == "Assistant" and _is_severely_repetitive(text):
            config.custom_print("Error", f"Blocked repetitive assistant output from chat log: {text[:80]}...")
            entry = f"{ts} [{role}] [repetitive output discarded]"
        self.chat_log.append(entry)
        if len(self.chat_log) > config.MAX_CHAT_LOG_ENTRIES:
            self.chat_log.pop(0)