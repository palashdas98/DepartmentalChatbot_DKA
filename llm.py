import os

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found in Environment Variables"
    )

client = Groq(
    api_key=GROQ_API_KEY
)

MODEL_NAME = "openai/gpt-oss-20b"


class LLMWrapper:

    SYSTEM_PROMPT = """
You are Tata Motors Department Knowledge Assistant.

Rules:

1. Use ONLY the supplied context.
2. Never hallucinate.
3. Give exact values from documents, copied exactly
   (same digits, same units) — never round or shorten a value.
4. Extract data correctly from tables.
5. When the context contains multiple related data points
   (e.g. several conditions, speeds, or gears), report ALL of
   them — completeness matters more than brevity here. Keep the
   answer direct and free of unnecessary commentary, but do not
   omit a data point just to keep the answer short.
6. If information is not available, reply exactly:

Information not found in documents.
"""

    def _call_groq(self, prompt, use_reasoning_effort):
        # openai/gpt-oss-20b is a REASONING model: part of its
        # token budget is spent on internal "thinking" tokens
        # (returned separately, not in .content) before it writes
        # the visible answer. Groq's own docs warn their default
        # of 1024 "may be too low for complex reasoning." A
        # previous fix here dropped this to 900 to dodge a TPM
        # rate-limit error, which starved the model of any tokens
        # left to write an actual answer after reasoning — that's
        # what caused "Relevant documents were retrieved but no
        # answer could be generated" to start appearing.
        # reasoning_effort="low" tells the model to spend only a
        # small number of tokens on reasoning (this task is
        # extraction/formatting, not deep analysis), which —
        # combined with max_tokens below — keeps real headroom for
        # the answer while still fitting Groq's free-tier 8000 TPM
        # cap alongside MAX_CONTEXT_CHARS in app.py/api.py. If you
        # raise MAX_CONTEXT_CHARS there, lower max_tokens here to
        # compensate (see that file's comment for the budget math).
        kwargs = dict(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": self.SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            max_tokens=2500
        )

        if use_reasoning_effort:
            kwargs["reasoning_effort"] = "low"

        return client.chat.completions.create(**kwargs)

    def invoke(self, prompt):

        try:

            try:
                response = self._call_groq(
                    prompt,
                    use_reasoning_effort=True
                )
            except TypeError:
                # Older installed `groq` SDK versions don't yet
                # know the reasoning_effort parameter and raise a
                # TypeError for an unexpected keyword argument.
                # Retry once without it rather than failing the
                # whole request — max_tokens=2500 alone still
                # avoids the empty-answer regression described
                # above, just without the extra reasoning-budget
                # control. Run `pip install -U groq` to get
                # reasoning_effort support.
                response = self._call_groq(
                    prompt,
                    use_reasoning_effort=False
                )

            content = response.choices[0].message.content

            if content is None:
                # Defensive: if the model still returns no content
                # (e.g. it used its entire budget on reasoning
                # despite reasoning_effort="low"), return an empty
                # string rather than crashing on .strip() — the
                # caller (app.py/api.py) already handles an empty
                # answer by showing a clear fallback message.
                return ""

            return content.strip()

        except Exception as e:

            print("\n========== GROQ ERROR ==========")
            print(str(e))
            print("================================\n")

            raise Exception(
                f"Groq API Error: {str(e)}"
            )


def get_llm():
    return LLMWrapper()
