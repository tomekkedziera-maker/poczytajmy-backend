# 1) Wklej plik
Set-Content -Path .\index.js -Value (Get-Clipboard) -Encoding UTF8

# 2) Opcjonalnie ustaw preferencję w Render (ENV):
# LLM_PREF = openai-first

# 3) Commit & push
git add index.js
git commit -m "feat: OpenAI-first with Groq fallback (LLM_PREF); bump 1.15"
git push origin main

