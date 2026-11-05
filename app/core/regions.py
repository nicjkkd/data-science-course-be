"""Region slugs shared with the frontend (web/src/lib/regions.ts). Labels are Ukrainian."""

REGIONS: dict[str, str] = {
    "vinnytsia": "Вінницька",
    "volyn": "Волинська",
    "dnipro": "Дніпропетровська",
    "donetsk": "Донецька",
    "zhytomyr": "Житомирська",
    "zakarpattia": "Закарпатська",
    "zaporizhzhia": "Запорізька",
    "ivano_frankivsk": "Івано-Франківська",
    "kyiv": "Київська",
    "kyiv_city": "м. Київ",
    "kirovohrad": "Кіровоградська",
    "luhansk": "Луганська",
    "lviv": "Львівська",
    "mykolaiv": "Миколаївська",
    "odesa": "Одеська",
    "poltava": "Полтавська",
    "rivne": "Рівненська",
    "sumy": "Сумська",
    "ternopil": "Тернопільська",
    "kharkiv": "Харківська",
    "kherson": "Херсонська",
    "khmelnytskyi": "Хмельницька",
    "cherkasy": "Черкаська",
    "chernivtsi": "Чернівецька",
    "chernihiv": "Чернігівська",
    "crimea": "АР Крим",
    "sevastopol": "м. Севастополь",
}

# Stable row order for region heatmaps (west → east, roughly as on the map).
REGION_ORDER: list[str] = [
    "zakarpattia", "lviv", "volyn", "ivano_frankivsk", "ternopil", "chernivtsi", "rivne",
    "khmelnytskyi", "zhytomyr", "vinnytsia", "kyiv", "kyiv_city", "chernihiv", "cherkasy",
    "kirovohrad", "odesa", "mykolaiv", "poltava", "sumy", "kharkiv", "dnipro", "kherson",
    "zaporizhzhia", "donetsk", "luhansk", "crimea", "sevastopol",
]  # fmt: skip
