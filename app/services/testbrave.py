import requests

response = requests.get(
  "https://api.search.brave.com/res/v1/web/search",
  headers={
    "Accept": "application/json",
    "Accept-Encoding": "gzip",
    "x-subscription-token": "BSALixqyj1Y0VfVuwME0iRYHFgysNeo"
  },
  params={
    "q": "SBI住信ネット銀行 顧客基盤 成長戦略 2025",
    "country": "JP",
    "search_lang": "jp",
    "ui_lang": "ja-JP"
  },
).json()