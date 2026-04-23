import os, asyncio, aiohttp, json
from dotenv import load_dotenv

load_dotenv()

GEMINI_KEY     = os.getenv("GEMINI_API_KEY")
GROQ_KEY       = os.getenv("GROQ_API_KEY")
COHERE_KEY     = os.getenv("COHERE_API_KEY")
HF_KEY         = os.getenv("HF_API_KEY")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")

# ── Shared prompt builder ──────────────────────────────────────
def build_prompt(acne_type: str, severity: str, tags: list) -> str:
    return f"""You are a professional dermatologist AI assistant.
A patient has been diagnosed with {acne_type.replace('_', ' ')} acne of {severity} severity.
Recommended ingredient tags: {', '.join(tags)}

Provide a structured skincare routine with these exact sections:
1. Morning Routine (step by step)
2. Evening Routine (step by step)
3. Key Ingredients to Use
4. Ingredients to Avoid
5. Lifestyle Tips

Be specific, practical, and concise. No disclaimers."""

# ── Individual providers ───────────────────────────────────────
        
async def call_groq(prompt: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 800
    }
    async with aiohttp.ClientSession() as s:
        async with s.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as r:
            data = await r.json()
            return data["choices"][0]["message"]["content"]

async def call_gemini(prompt: str) -> str:
    # ✅ Confirmed working model from Google API docs
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    async with aiohttp.ClientSession() as s:
        async with s.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as r:
            data = await r.json()
            candidates = data.get("candidates", [])
            if candidates:
                return candidates[0]["content"]["parts"][0]["text"]
            raise Exception(f"Gemini error: {data}")

async def call_cohere(prompt: str) -> str:
    url = "https://api.cohere.com/v2/chat"
    headers = {
        "Authorization": f"Bearer {COHERE_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "command-r-plus",
        "messages": [{"role": "user", "content": prompt}]
    }
    async with aiohttp.ClientSession() as s:
        async with s.post(url, headers=headers, json=payload,
                          timeout=aiohttp.ClientTimeout(total=30)) as r:
            data = await r.json()
            # Cohere v2 returns content as a plain string directly
            message = data.get("message", {})
            content = message.get("content", "")
            if isinstance(content, str) and content:
                return content
            elif isinstance(content, list) and content:
                first = content[0]
                if isinstance(first, dict):
                    return first.get("text", str(first))
                return str(first)
            raise Exception(f"Cohere unexpected structure: {data}")

async def call_huggingface(prompt: str) -> str:
    # ✅ HF updated to new router endpoint (OpenAI compatible)
    url = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {HF_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "messages": [
            {"role": "system", "content": "You are a professional dermatologist assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 600
    }
    async with aiohttp.ClientSession() as s:
        async with s.post(url, headers=headers, json=payload,
                          timeout=aiohttp.ClientTimeout(total=60)) as r:
            if r.status == 503:
                raise Exception("HF model loading, retry in 20s")
            data = await r.json()
            if "choices" in data:
                return data["choices"][0]["message"]["content"]
            raise Exception(f"HF error: {data}")

async def call_openrouter(prompt: str) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://dermascan.app",
        "X-Title": "DermaScan"
    }
    payload = {
        # ✅ Confirmed free model on OpenRouter
        "model": "google/gemma-3-4b-it:free",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 800
    }
    async with aiohttp.ClientSession() as s:
        async with s.post(url, headers=headers, json=payload,
                          timeout=aiohttp.ClientTimeout(total=30)) as r:
            data = await r.json()
            if "choices" in data:
                return data["choices"][0]["message"]["content"]
            raise Exception(f"OpenRouter error: {data}")

# ── Provider map ───────────────────────────────────────────────
PROVIDERS = {
    "gemini":      call_gemini,
    "groq":        call_groq,
    "cohere":      call_cohere,
    "huggingface": call_huggingface,
    "openrouter":  call_openrouter,
}

# ── Combined: pick longest response ───────────────────────────
async def call_combined(prompt: str) -> str:
    tasks = [fn(prompt) for fn in PROVIDERS.values()]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    valid = [r for r in results if isinstance(r, str) and len(r) > 100]
    if not valid:
        raise Exception("All providers failed")
    return max(valid, key=len)

# ── Consensus: merge all responses ────────────────────────────
async def call_consensus(prompt: str) -> str:
    tasks = [fn(prompt) for fn in PROVIDERS.values()]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    valid = [r for r in results if isinstance(r, str) and len(r) > 100]
    if not valid:
        raise Exception("All providers failed")

    merged = "🔬 CONSENSUS SKINCARE ROUTINE (synthesized from multiple AI models)\n\n"
    provider_names = list(PROVIDERS.keys())
    for i, result in enumerate(valid):
        merged += f"--- {provider_names[i].upper()} RECOMMENDATION ---\n{result}\n\n"
    return merged

# ── Main entry point ───────────────────────────────────────────
async def get_skincare_routine(
    acne_type: str,
    severity: str,
    tags: list,
    provider: str = "gemini"
) -> dict:
    prompt = build_prompt(acne_type, severity, tags)

    try:
        if provider == "combined":
            response = await call_combined(prompt)
        elif provider == "consensus":
            response = await call_consensus(prompt)
        elif provider in PROVIDERS:
            response = await PROVIDERS[provider](prompt)
        else:
            raise ValueError(f"Unknown provider: {provider}")

        return {"provider": provider, "routine": response, "status": "success"}

    except Exception as e:
        # Auto-fallback to gemini if chosen provider fails
        try:
            fallback = await call_gemini(prompt)
            return {"provider": "gemini_fallback", "routine": fallback, "status": "fallback"}
        except:
            return {"provider": "none", "routine": str(e), "status": "error"}