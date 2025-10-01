# 1) Wejdź do katalogu backendu
cd D:\poczytajmy-backend

# 2) (opcjonalnie) utwórz branch
git checkout -b fix/comprehend-priority

# 3) Wklej nową zawartość index.js z CLIPBOARD (skopiuj cały plik, potem:)
Set-Content -Path .\index.js -Value (Get-Clipboard) -Encoding UTF8

# 4) Commit + push
git add .\index.js
git commit -m "comprehend: priorytet dopełnienia (Co/Czego/W co) w 1. os.; doprecyzowany prompt 3. os."
git push origin HEAD

