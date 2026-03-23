import os
from dotenv import load_dotenv
load_dotenv()
import json
import logging
import unicodedata
from enum import Enum


from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from openai import OpenAI, APIConnectionError, APIStatusError, RateLimitError

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Message Fixer API",
    description="Rewrites messages in Turkish or English with a chosen tone.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

_api_key = os.getenv("GROQ_API_KEY")
if not _api_key:
    raise RuntimeError("GROQ_API_KEY environment variable is not set.")

client = OpenAI(
    api_key=_api_key,
    base_url="https://api.groq.com/openai/v1"
)


# ── Enums & constants ──────────────────────────────────────────────────────────
class Tone(str, Enum):
    professional = "professional"
    friendly     = "friendly"
    polite       = "polite"
    shorter      = "shorter"
    more_formal  = "more formal"


MAX_INPUT_CHARS = 2000
MODEL = "llama-3.1-8b-instant"

# Turkish-exclusive Unicode characters — presence is a definitive signal
TURKISH_CHARS = set("çÇğĞşŞıİöÖüÜ")

# High-frequency Turkish function words — density signals Turkish even without
# special characters (e.g. messages typed on a keyboard lacking Turkish layout)
TURKISH_FUNCTION_WORDS = {
    "bir", "bu", "şu", "o", "ve", "ile", "için", "gibi", "ama", "fakat",
    "ancak", "da", "de", "ki", "mi", "mı", "mu", "mü", "ne", "nasıl",
    "neden", "çok", "daha", "en", "biz", "siz", "onlar", "ben", "sen",
    "var", "yok", "olan", "ise", "bile", "her", "hiç", "hem",
    "ya", "veya", "bana", "sana", "beni", "seni", "bunu", "şunu",
    "merhaba", "teşekkür", "ederim", "lütfen", "tamam", "evet", "hayır",
}


# ── Language detection (deterministic pre-pass before hitting the LLM) ────────
def detect_language_heuristic(text: str) -> str | None:
    """
    Returns 'Turkish' if confident, else None (defer to LLM).

    Three independent signals — any strong one wins:
      1. Turkish-exclusive Unicode characters (ğ, ş, ı, İ, ö, ü, ç)
      2. Turkish function-word density
      3. Falls through to None → LLM decides
    """
    text_nfc = unicodedata.normalize("NFC", text)

    # Signal 1: even one unambiguous Turkish character is highly reliable
    turkish_char_count = sum(1 for ch in text_nfc if ch in TURKISH_CHARS)
    if turkish_char_count >= 1:
        logger.debug("Heuristic → Turkish (special chars: %d)", turkish_char_count)
        return "Turkish"

    # Signal 2: ≥2 Turkish function words in the token set
    tokens = set(text_nfc.lower().split())
    overlap = tokens & TURKISH_FUNCTION_WORDS
    if len(overlap) >= 2:
        logger.debug("Heuristic → Turkish (function words: %s)", overlap)
        return "Turkish"

    return None  # not confident; let the model decide


def normalize_input(text: str) -> str:
    """
    NFC-normalize so composed Turkish characters (ş, ç, ğ …) are in their
    canonical form. Prevents tokeniser from splitting decomposed combining chars.
    """
    return unicodedata.normalize("NFC", text)


# ── Tone profiles (bilingual) ──────────────────────────────────────────────────
# Every tone has both an English and a Turkish instruction block.
# The model receives only the relevant block (or both for ambiguous input).
TONE_PROFILES: dict[Tone, dict] = {
    Tone.professional: {
        "en": {
            "description": "Clear, confident, and business-appropriate.",
            "do":    "Use precise vocabulary. Be direct. Structure sentences logically.",
            "avoid": "Slang, filler words, excessive hedging, overly casual contractions.",
        },
        "tr": {
            "description": "Açık, özgüvenli ve iş ortamına uygun.",
            "do":    "Kesin ve doğrudan bir dil kullan. Cümleleri mantıklı kur. Ç, ş, ğ, ı, ö, ü harflerini her zaman doğru yaz.",
            "avoid": "Argo, dolgu sözcükler, aşırı gayri resmi kısaltmalar.",
        },
    },
    Tone.friendly: {
        "en": {
            "description": "Warm, approachable, and genuinely human.",
            "do":    "Use natural contractions. Show personality. Keep it conversational.",
            "avoid": "Stiff phrasing, corporate language, form-letter tone.",
        },
        "tr": {
            "description": "Sıcak, samimi ve doğal bir Türkçe.",
            "do":    "Günlük konuşma diline uygun, içten bir üslup kullan. Kişilik yansıt.",
            "avoid": "Resmi bürokratik dil, soğuk ifadeler, kalıp cümleler.",
        },
    },
    Tone.polite: {
        "en": {
            "description": "Respectful and courteous without being cold.",
            "do":    "Soften requests gracefully. Acknowledge the other person.",
            "avoid": "Bluntness, passive aggression, excessive formality.",
        },
        "tr": {
            "description": "Saygılı ve nazik — ama soğuk değil.",
            "do":    "İstekleri kibar bir şekilde ifade et. 'Lütfen', 'rica ederim', 'teşekkür ederim' gibi ifadeleri doğal kullan.",
            "avoid": "Kaba ifadeler, pasif agresif ton, insanı uzaklaştıran aşırı resmiyet.",
        },
    },
    Tone.shorter: {
        "en": {
            "description": "Concise — every word earns its place.",
            "do":    "Cut filler and redundancy. Lead with the main point.",
            "avoid": "Preambles, throat-clearing phrases, restating what was just said.",
        },
        "tr": {
            "description": "Kısa ve öz — her kelime yerli yerinde.",
            "do":    "Gereksiz kelime ve tekrarları çıkar. Ana noktayla başla.",
            "avoid": "Uzun girişler, tekrar eden ifadeler, gereksiz açıklamalar.",
        },
    },
    Tone.more_formal: {
        "en": {
            "description": "Elevated register for official or high-stakes communication.",
            "do":    "Use full grammatical forms. Prefer sophisticated vocabulary where natural.",
            "avoid": "Contractions, idioms, slang, casual punctuation.",
        },
        "tr": {
            "description": "Resmi yazışmalara uygun yüksek register.",
            "do":    "Tam dilbilgisel formlar kullan. Saygı eklerini doğru uygula (-iniz/-ınız, Sayın). Türkçe noktalamayı doğru yaz.",
            "avoid": "Kısaltmalar, argo, günlük konuşma dili, fazla samimi ifadeler.",
        },
    },
}


# ── Prompt builder ─────────────────────────────────────────────────────────────
def build_prompt(text: str, tone: Tone, hint: str | None) -> str:
    """
    Constructs a bilingual-aware prompt.

    hint = "Turkish"  → model is told the language is Turkish; Turkish profile only
    hint = None       → model detects language; both profiles provided
    """
    profile = TONE_PROFILES[tone]

    if hint == "Turkish":
        language_block = (
            "LANGUAGE: The input is TURKISH.\n"
            "→ You MUST rewrite in Turkish.\n"
            "→ Do NOT translate to English under any circumstances.\n"
            '→ Set "language" to "Turkish" in your JSON response.'
        )
        tone_block = (
            f"TONE — {tone.value.upper()} (Turkish):\n"
            f"  Hedef:   {profile['tr']['description']}\n"
            f"  Yap:     {profile['tr']['do']}\n"
            f"  Yapma:   {profile['tr']['avoid']}"
        )
    else:
        language_block = (
            "LANGUAGE: Detect whether the message is Turkish or English.\n"
            "→ Rewrite in that SAME language. Never translate.\n"
            '→ Set "language" to "Turkish" or "English" accordingly.'
        )
        tone_block = (
            f"TONE — {tone.value.upper()} (if English):\n"
            f"  Goal:    {profile['en']['description']}\n"
            f"  Do:      {profile['en']['do']}\n"
            f"  Avoid:   {profile['en']['avoid']}\n\n"
            f"TONE — {tone.value.upper()} (if Turkish):\n"
            f"  Hedef:   {profile['tr']['description']}\n"
            f"  Yap:     {profile['tr']['do']}\n"
            f"  Yapma:   {profile['tr']['avoid']}"
        )

    return f"""You are an expert bilingual writing coach, fluent in both Turkish and English.
Your only job: rewrite the message so it sounds polished and natural to a native speaker.
You never translate. You never explain. You only rewrite.

━━━ {language_block}

━━━ {tone_block}

━━━ UNIVERSAL RULES (apply regardless of language):
1. MEANING       Keep the full original intent. Do not add or remove information.
2. GRAMMAR       Fix all grammar, spelling, and punctuation silently.
3. NATURALNESS   Output must read like a thoughtful native speaker wrote it.
                 Vary sentence length and structure. No robotic repetition.
4. TURKISH RULES (enforce when language is Turkish):
   a. Always write special characters correctly: ç ğ ş ı İ ö ü — never substitute
      (no "i" for "ı", no "g" for "ğ", no "s" for "ş", no "c" for "ç").
   b. Apply correct vowel harmony in all suffixes.
   c. Use the right formality register: sen/siz, -iyor/-mekte, -dır/-dir as tone demands.
   d. Avoid English-calque constructions; use natural Turkish phrasing and word order (SOV).
5. LENGTH        Match the original length unless tone is "shorter".
6. OUTPUT        Return ONLY a raw JSON object — no markdown, no labels, no commentary.
                 Required schema: {{"language": "Turkish" | "English", "improved": "<text>"}}

━━━ MESSAGE TO REWRITE:
{text}"""


# ── OpenAI call ────────────────────────────────────────────────────────────────
def call_openai(prompt: str) -> dict:
    """Send prompt to OpenAI and return the parsed JSON result."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,          # lower = more faithful to instructions
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
    except RateLimitError as e:
        logger.warning("Rate limit: %s", e)
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                            detail="Rate limit reached. Please wait and try again.")
    except APIConnectionError as e:
        logger.error("Connection error: %s", e)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Could not connect to the AI service.")
    except APIStatusError as e:
        logger.error("API error %s: %s", e.status_code, e.message)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY,
                            detail=f"AI service error: {e.message}")

    raw = response.choices[0].message.content or ""

    try:
        parsed = json.loads(raw)
    except ValueError:
        logger.error("Non-JSON response: %r", raw)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Unexpected response from AI service.")

    if "improved" not in parsed or "language" not in parsed:
        logger.error("Missing keys in response: %r", parsed)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="AI response was missing required fields.")

    return parsed


def _turkish_output_looks_valid(text: str) -> bool:
    """
    Lightweight sanity check for Turkish output.
    Returns False if the result looks like plain ASCII English
    when we expected Turkish — used only for logging.
    """
    tr_chars = sum(1 for c in text if c in TURKISH_CHARS)
    tokens   = set(text.lower().split())
    return tr_chars > 0 or bool(tokens & TURKISH_FUNCTION_WORDS)


# ── Schemas ────────────────────────────────────────────────────────────────────
class FixRequest(BaseModel):
    text: str
    tone: Tone

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text must not be empty or whitespace.")
        if len(v) > MAX_INPUT_CHARS:
            raise ValueError(f"text must not exceed {MAX_INPUT_CHARS} characters.")
        return v


class FixResponse(BaseModel):
    original:          str
    improved:          str
    tone:              Tone
    language:          str
    language_detected: str   # "heuristic" | "model" — useful for debugging


class ErrorResponse(BaseModel):
    detail: str


# ── Global error handler ───────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred. Please try again."},
    )


# ── Endpoint ───────────────────────────────────────────────────────────────────
@app.post(
    "/fix",
    response_model=FixResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        502: {"model": ErrorResponse, "description": "AI service error"},
        503: {"model": ErrorResponse, "description": "AI service unavailable"},
    },
    summary="Rewrite a message with the given tone",
)
async def fix_message(req: FixRequest):
    # Step 1 — normalise Unicode (NFC) so Turkish composed chars are canonical
    clean = normalize_input(req.text)

    # Step 2 — fast deterministic language hint (no API cost)
    hint   = detect_language_heuristic(clean)
    method = "heuristic" if hint else "model"
    logger.info("fix | tone=%s | chars=%d | lang_hint=%s", req.tone, len(clean), hint or "none")

    # Step 3 — build language-aware prompt and call the model
    prompt = build_prompt(clean, req.tone, hint)
    result = call_openai(prompt)

    # Step 4 — log a warning if Turkish output looks suspicious
    if result["language"] == "Turkish" and not _turkish_output_looks_valid(result["improved"]):
        logger.warning("Turkish output failed sanity check: %r", result["improved"])

    return FixResponse(
        original=req.text,
        improved=result["improved"],
        tone=req.tone,
        language=result["language"],
        language_detected=method,
    )


@app.get("/health", summary="Health check")
async def health():
    return {"status": "ok", "model": MODEL, "version": app.version}
