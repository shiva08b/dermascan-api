import os, asyncio, aiohttp, json, re
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


# ── JSON extraction helper ─────────────────────────────────────
def _extract_json(text: str) -> str:
    """Pull a JSON array/object out of text that may contain markdown fences."""
    # strip ```json ... ``` or ``` ... ```
    m = re.search(r'```(?:json)?\s*([\[{][\s\S]*?)\s*```', text, re.DOTALL)
    if m:
        return m.group(1)
    # bare JSON array
    m = re.search(r'(\[[\s\S]*\])', text)
    if m:
        return m.group(1)
    return text


# ── Products prompt ────────────────────────────────────────────
def build_products_prompt(acne_type: str, severity: str, skin_type: str, tags: list) -> str:
    tag_str = ', '.join(tags) if tags else 'general'
    return f"""You are a skincare product recommendation expert for the Indian market.
A patient has {acne_type.replace('_', ' ')} acne of {severity} severity with {skin_type} skin type.
AI analysis ingredient tags: {tag_str}

Return ONLY a valid JSON array of exactly 5 product objects. No explanation, no markdown code fences, just the raw JSON array.

Each object MUST follow this exact structure:
{{
  "id": "ai-prod-1",
  "name": "Actual product name",
  "brand": "Real brand name",
  "category": "Cleanser",
  "sponsored": false,
  "price": 599,
  "rating": 4.5,
  "offer": "Brief offer description",
  "retailer": "Nykaa",
  "skinTypes": ["{skin_type}"],
  "concernTags": ["salicylic_acid"],
  "description": "One sentence explaining why this product helps this acne type."
}}

Rules:
- Use real, well-known brands: CeraVe, Minimalist, The Ordinary, Biotique, Himalaya, Mamaearth, Dot & Key, Plum, Cetaphil, Neutrogena, La Roche-Posay, Reequil, The Inkey List, COSRX, Acne-Aid
- Exactly 2 products must have "sponsored": true, 3 must have "sponsored": false
- Categories: Cleanser, Serum, Moisturizer, Sunscreen, Toner, Treatment
- Retailers: Amazon, Nykaa, Flipkart, Tira, Sephora, Myntra, Kindlife
- Price in INR only, range 150-2500
- IDs must be: ai-prod-1, ai-prod-2, ai-prod-3, ai-prod-4, ai-prod-5"""


# ── Remedies prompt ────────────────────────────────────────────
def build_remedies_prompt(acne_type: str, skin_type: str) -> str:
    return f"""You are a dermatologist specializing in evidence-based home remedies for Indian skin types.
A patient has {acne_type.replace('_', ' ')} acne with {skin_type} skin type.

Return ONLY a valid JSON array of exactly 3 home remedy objects. No explanation, no markdown code fences, just the raw JSON array.

Each object MUST follow this exact structure:
{{
  "id": "ai-remedy-1",
  "title": "Descriptive remedy title",
  "bestFor": ["{skin_type}"],
  "ingredients": ["2 tsp honey", "1 tbsp yogurt"],
  "steps": ["Step 1 detail", "Step 2 detail", "Step 3 detail"],
  "usage": "How often and when to apply",
  "precautions": "One important precaution or contraindication"
}}

Rules:
- Use ingredients commonly available in Indian households
- Each remedy must have 3-5 ingredients and exactly 3 steps
- Tailor specifically to {acne_type.replace('_', ' ')} acne on {skin_type} skin
- Include measurement quantities in ingredients
- IDs must be: ai-remedy-1, ai-remedy-2, ai-remedy-3"""


# ── AI products generator ──────────────────────────────────────
async def get_ai_products(
    acne_type: str,
    severity: str,
    skin_type: str,
    tags: list,
    provider: str = "gemini"
) -> list:
    prompt = build_products_prompt(acne_type, severity, skin_type, tags)
    async def _parse(text: str) -> list:
        data = json.loads(_extract_json(text))
        return data if isinstance(data, list) else []

    try:
        fn = PROVIDERS.get(provider, call_gemini)
        return await _parse(await fn(prompt))
    except Exception:
        try:
            return await _parse(await call_gemini(prompt))
        except Exception:
            return []


# ── AI remedies generator ──────────────────────────────────────
async def get_ai_remedies(
    acne_type: str,
    skin_type: str,
    provider: str = "gemini"
) -> list:
    prompt = build_remedies_prompt(acne_type, skin_type)
    async def _parse(text: str) -> list:
        data = json.loads(_extract_json(text))
        return data if isinstance(data, list) else []

    try:
        fn = PROVIDERS.get(provider, call_gemini)
        return await _parse(await fn(prompt))
    except Exception:
        try:
            return await _parse(await call_gemini(prompt))
        except Exception:
            return []