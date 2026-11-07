"""
Smartphone assistant: the logic of notebook 08 as an importable module.

Pipeline:
    sentence -> keyword extraction (+ optional Ollama model for paraphrases)
             -> PhoneRequest (validated)
             -> rules: filter, top-15% shortlist (at least 5), best value (at most 5)

The rules, keyword lists, and model check are the same as in
notebooks/08_natural_language_recommendations.ipynb.
"""

from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
import re
from typing import Literal

import duckdb
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, model_validator

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "phones_with_value_scores.csv"


# =========================================================================== data

def extract_zoom(camera_label):

    if isinstance(camera_label, str) and camera_label.startswith("Telephoto"):
        match = re.search(r"(\d+(?:\.\d+)?)x", camera_label)
        if match:
            return float(match.group(1))

    return np.nan


def load_data(path=DATA_PATH):

    data = pd.read_csv(path, parse_dates=["launch_date"])

    # Telephoto zoom from the third and fourth camera labels
    data["telephoto_zoom"] = pd.concat(
        [data["third"].map(extract_zoom), data["fourth"].map(extract_zoom)],
        axis=1
    ).max(axis=1)

    # Sensor information: recorded, or assumed for Samsung, Apple, and Google
    data["camera_sensor_known"] = (
        data["camera_sensor_name"].notna()
        | data["brand"].isin(SENSOR_INFO_BRANDS)
    ).astype(int)

    # Peak brightness, missing values filled with the display-type mean
    hbm_mean_by_display = data.groupby("display_type")["hbm"].mean()
    data["hbm_filled"] = data["hbm"].fillna(data["display_type"].map(hbm_mean_by_display))

    # Brand value: target encoding of brand with price
    data["brand_value"] = data["brand"].map(data.groupby("brand")["price"].mean())

    return data


SENSOR_INFO_BRANDS = ["Samsung", "Apple", "Google"]

data = load_data()

_connection = duckdb.connect()
_connection.register("phones_frame", data)
_connection.execute("CREATE TABLE phones AS SELECT * FROM phones_frame")


def query(sql, params=()):
    """Run a query on a cursor, so the app can serve several sessions safely."""
    return _connection.cursor().execute(sql, list(params)).df()


VOCABULARY_COLUMNS = ["brand", "processor_type", "display_type", "speaker", "frame_material", "back_material", "ip_rating"]

VOCABULARY = {
    column: sorted(data[column].dropna().unique().tolist())
    for column in VOCABULARY_COLUMNS
}


# =========================================================================== requirements and features

REQUIREMENTS = {
    "stereo_speakers": {
        "description": "Good sound: stereo speakers",
        "sql": "speaker = 'Stereo'"
    },
    "waterproof": {
        "description": "Waterproof: IP68 or IP69 rating",
        "sql": "ip_rating IN (68, 69)"
    },
    "good_build": {
        "description": "Good build: protection glass recorded, frame not plastic, back not plastic",
        "sql": "protection IS NOT NULL AND frame_material <> 'Plastic' AND back_material <> 'Plastic'"
    },
    "wireless_charging": {
        "description": "Supports wireless charging",
        "sql": "wireless_charging"
    }
}

FEATURES = {
    "battery": {"rank_by": [("battery", "higher")], "description": "Battery capacity"},
    "performance": {"rank_by": [("antutu", "higher")], "description": "AnTuTu benchmark score"},
    "camera": {
        "rank_by": [("useful_camera_number", "higher"), ("camera_sensor_known", "higher")],
        "description": "Useful cameras first, then documented sensor information"
    },
    "charging": {"rank_by": [("charging_speed", "higher")], "description": "Wired charging speed"},
    "brightness": {"rank_by": [("hbm", "higher")], "description": "Peak display brightness"},
    "refresh_rate": {"rank_by": [("refresh_rate", "higher")], "description": "Display refresh rate"},
    "slimness": {"rank_by": [("thickness", "lower")], "description": "Body thickness"},
    "lightness": {"rank_by": [("weight", "lower")], "description": "Weight"},
    "compactness": {"rank_by": [("size", "lower")], "description": "Smallest screen size"},
    "zoom": {"rank_by": [("telephoto_zoom", "higher")], "description": "Telephoto zoom factor"},
    "multimedia": {
        "rank_by": [("size", "higher"), ("ppi", "higher"), ("hbm_filled", "higher")],
        "requires": ["stereo_speakers"],
        "description": "Largest screen, then sharpest, then brightest; stereo speakers required"
    },
    "brand_value": {"rank_by": [("brand_value", "higher")], "description": "Brand's mean price (target encoding)"},
}


# =========================================================================== structured request

Brand = Literal[tuple(VOCABULARY["brand"])]
ProcessorType = Literal[tuple(VOCABULARY["processor_type"])]
DisplayType = Literal[tuple(VOCABULARY["display_type"])]
Requirement = Literal[tuple(REQUIREMENTS)]
Feature = Literal[tuple(FEATURES)]


class PhoneRequest(BaseModel):

    max_price: int | None = None
    min_price: int | None = None
    around_price: int | None = None

    brands: list[Brand] = Field(default_factory=list)
    exclude_brands: list[Brand] = Field(default_factory=list)
    processor_types: list[ProcessorType] = Field(default_factory=list)
    display_types: list[DisplayType] = Field(default_factory=list)

    requirements: list[Requirement] = Field(default_factory=list)

    rank_feature: Feature | None = None
    other_features: list[Feature] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_prices(self):

        for name in ["max_price", "min_price", "around_price"]:
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")

        if self.min_price and self.max_price and self.min_price > self.max_price:
            raise ValueError("min_price is greater than max_price")

        return self


# =========================================================================== rules

TOP_FRACTION = 0.15
MAX_RESULTS = 5
BUDGET_FLOOR = 0.80       # "under 60k"  -> 48,000 to 60,000
AROUND_TOLERANCE = 0.10   # "around 30k" -> 27,000 to 33,000
DISCOUNT_STRETCH = 1.10   # "under 40k" also considers phones up to 44,000, tagged


def top_fraction_shortlist(phones, rank_by, top_fraction=TOP_FRACTION, min_size=MAX_RESULTS):

    columns = [column for column, _ in rank_by]
    ascending = [direction == "lower" for _, direction in rank_by]

    rankable = phones.dropna(subset=columns)

    if rankable.empty:
        return rankable

    ranked = rankable.sort_values(columns, ascending=ascending)

    # Phones with identical values on every ranking column share a tie group (1 = best)
    tie_group = ranked[columns].ne(ranked[columns].shift()).any(axis=1).cumsum()

    # Top 15%, but never fewer than min_size phones (or all of them, if fewer match)
    shortlist_size = max(ceil(top_fraction * len(ranked)), min(min_size, len(ranked)))
    cutoff_group = tie_group.iloc[shortlist_size - 1]

    return ranked[tie_group <= cutoff_group]


def price_range(request):

    min_price, max_price = request.min_price, request.max_price

    if request.around_price is not None:
        min_price = min_price or round(request.around_price * (1 - AROUND_TOLERANCE))
        max_price = max_price or round(request.around_price * (1 + AROUND_TOLERANCE))

    # "Under X" with no lower limit: search the price class 0.8X to X
    if max_price is not None and min_price is None:
        min_price = round(max_price * BUDGET_FLOOR)

    return min_price, max_price


def discount_ceiling(request):
    """Highest price searched. For "under X", phones up to 1.1X are also considered."""

    min_price, max_price = price_range(request)

    is_under_request = (
        request.max_price is not None
        and request.min_price is None
        and request.around_price is None
    )

    if is_under_request:
        return round(max_price * DISCOUNT_STRETCH)

    return max_price


def price_text(min_price, max_price, ceiling=None):

    if min_price and max_price:
        text = f"Rs. {min_price:,} to Rs. {max_price:,}"
    elif min_price:
        text = f"from Rs. {min_price:,}"
    elif max_price:
        text = f"up to Rs. {max_price:,}"
    else:
        text = "any price"

    if ceiling and max_price and ceiling > max_price:
        text += f"; up to Rs. {ceiling:,} with possible discounts"

    return text


def budget_text(request):
    return price_text(*price_range(request), discount_ceiling(request))


def all_requirements(request):

    feature = FEATURES.get(request.rank_feature) if request.rank_feature else None
    extra = feature.get("requires", []) if feature else []

    return list(dict.fromkeys(request.requirements + extra))


def find_candidates(request, skip=()):
    """Run the filter query. `skip` names conditions to leave out (used to explain empty results)."""

    min_price, max_price = price_range(request)

    conditions = ["availability = 1"]
    params = []

    if "budget" not in skip:
        ceiling = discount_ceiling(request)
        if ceiling is not None:
            conditions.append("price <= ?")
            params.append(ceiling)
        if min_price is not None:
            conditions.append("price >= ?")
            params.append(min_price)

    for name, column, values, operator in [
        ("brands", "brand", request.brands, "IN"),
        ("exclude_brands", "brand", request.exclude_brands, "NOT IN"),
        ("processor_types", "processor_type", request.processor_types, "IN"),
        ("display_types", "display_type", request.display_types, "IN"),
    ]:
        if values and name not in skip:
            placeholders = ", ".join("?" for _ in values)
            conditions.append(f"{column} {operator} ({placeholders})")
            params.extend(values)

    for requirement in all_requirements(request):
        if requirement not in skip:
            conditions.append(f"({REQUIREMENTS[requirement]['sql']})")

    return query(f"SELECT * FROM phones WHERE {' AND '.join(conditions)}", params)


def explain_no_match(request):

    min_price, max_price = price_range(request)

    labels = {
        "budget": f"the budget ({budget_text(request)})",
        "brands": f"the brand choice ({', '.join(request.brands)})",
        "exclude_brands": f"the excluded brands ({', '.join(request.exclude_brands)})",
        "processor_types": f"the processor choice ({', '.join(request.processor_types)})",
        "display_types": f"the display choice ({', '.join(request.display_types)})",
    }
    labels.update({
        requirement: f"the requirement '{REQUIREMENTS[requirement]['description']}'"
        for requirement in all_requirements(request)
    })

    active = [
        name for name in labels
        if (name == "budget" and (min_price or max_price))
        or (name in ("brands", "exclude_brands", "processor_types", "display_types") and getattr(request, name))
        or name in all_requirements(request)
    ]

    lines = ["No available phone matches all of these requirements."]

    for name in active:
        matches = len(find_candidates(request, skip=(name,)))
        if matches:
            lines.append(f"- Without {labels[name]}, {matches} phones would match.")

    if request.brands:
        brand_phones = data[(data["availability"] == 1) & data["brand"].isin(request.brands)]
        if not brand_phones.empty:
            lines.append(
                f"- Available {', '.join(request.brands)} phones cost "
                f"Rs. {brand_phones['price'].min():,} to Rs. {brand_phones['price'].max():,}."
            )

    if len(lines) == 1:
        lines.append("- Several conditions together rule out every phone; try relaxing more than one.")

    return "\n".join(lines)


@dataclass
class Recommendation:
    table: pd.DataFrame
    summary: str = ""
    no_match: str = ""
    rank_columns: list = field(default_factory=list)


def recommend(request):

    candidates = find_candidates(request)

    if candidates.empty:
        return Recommendation(
            table=pd.DataFrame(columns=["name", "brand", "price", "value_score"]),
            no_match=explain_no_match(request)
        )

    feature = FEATURES.get(request.rank_feature) if request.rank_feature else None

    # Shortlist for the ranking feature, or for the value score when no feature was requested
    if feature:
        rank_by = feature["rank_by"]
        rank_columns = [column for column, _ in rank_by]
        rank_ascending = [direction == "lower" for _, direction in rank_by]
    else:
        rank_by = [("value_score", "higher")]
        rank_columns, rank_ascending = [], []

    shortlist = top_fraction_shortlist(candidates, rank_by)

    # Best value, at most MAX_RESULTS
    result = (
        shortlist
        .sort_values(
            ["value_score"] + rank_columns + ["price"],
            ascending=[False] + rank_ascending + [True]
        )
        .head(MAX_RESULTS)
    )

    # Phones above the asked maximum are within reach only with a discount
    asked_max = price_range(request)[1]
    result = result.assign(note=[
        f"Could be available at Rs. {asked_max:,} with some discounts"
        if asked_max is not None and price > asked_max else ""
        for price in result["price"]
    ])

    ranked_on = request.rank_feature if feature else "value score"
    summary = (
        f"{len(candidates)} phones match the filters ({budget_text(request)}) "
        f"-> {len(shortlist)} shortlisted for {ranked_on} "
        f"-> {len(result)} recommended"
    )

    display_columns = ["name", "brand", "price"] + [
        column for column in rank_columns if column != "camera_sensor_known"
    ]

    table = (
        result[display_columns + ["value_score"] + (["note"] if result["note"].ne("").any() else [])]
        .round({"brand_value": 0, "hbm_filled": 0})
        .reset_index(drop=True)
    )

    return Recommendation(table=table, summary=summary, rank_columns=rank_columns)


# =========================================================================== keyword extraction

SUB_BRANDS = {
    "iphone": "Apple",
    "pixel": "Google",
    "galaxy": "Samsung",
    "redmi": "Xiaomi",
    "narzo": "Realme",
    "nord": "OnePlus",
    "one plus": "OnePlus",
    "moto": "Motorola",
    "cmf": "Nothing",
    "nothing phone": "Nothing",
}

# "Nothing" is also an ordinary English word ("nothing too expensive"),
# so the brand is only recognised as "Nothing Phone" or "CMF"
AMBIGUOUS_BRAND_WORDS = {"nothing"}

BRAND_ALIASES = {
    brand.lower(): brand
    for brand in VOCABULARY["brand"]
    if brand.lower() not in AMBIGUOUS_BRAND_WORDS
}
BRAND_ALIASES.update(SUB_BRANDS)

PROCESSOR_ALIASES = {processor.lower(): processor for processor in VOCABULARY["processor_type"]}
PROCESSOR_ALIASES.update({"dimensity": "Mediatek", "helio": "Mediatek", "bionic": "Apple Bionic"})

DISPLAY_ALIASES = {display.lower(): display for display in VOCABULARY["display_type"]}
DISPLAY_ALIASES.update({"oled": "Amoled", "poled": "Amoled", "p-oled": "Amoled", "lcd": "IPS LCD", "ips": "IPS LCD"})

REQUIREMENT_TERMS = {
    "stereo_speakers": ["stereo", "speaker", "speakers", "sound", "audio", "music", "loud"],
    "waterproof": [
        "waterproof", "water proof", "water-proof", "water resistant", "water-resistant",
        "water resistance", "rain", "rainproof", "pool", "swim", "swimming", "underwater", "ip68", "ip69"
    ],
    "good_build": [
        "build", "built", "build quality", "sturdy", "solid", "durable", "rugged", "metal",
        "plastic", "plasticky", "premium build", "premium feel", "feels premium"
    ],
    "wireless_charging": ["wireless charging", "wireless"],
}

FEATURE_TERMS = {
    "battery": [
        "battery", "batteries", "battery life", "backup", "long lasting", "long-lasting",
        "lasts", "mah", "endurance", "dies"
    ],
    "performance": [
        "performance", "gaming", "games", "game", "gamer", "powerful", "processor", "chipset",
        "lag", "lag-free", "antutu", "benchmark", "speed"
    ],
    "camera": [
        "camera", "cameras", "photo", "photos", "photography", "picture", "pictures", "pics",
        "selfie", "selfies", "low light", "low-light", "night photos", "instagram",
        "video recording", "record videos", "vlogging"
    ],
    "charging": [
        "charging", "charge", "charges", "charger", "fast charging", "quick charging",
        "charging speed", "watt", "watts"
    ],
    "brightness": ["bright", "brightness", "brightest", "sunlight", "outdoor", "outdoors", "nits"],
    "refresh_rate": ["refresh rate", "refresh", "smooth scrolling", "smooth display", "120hz", "144hz", "165hz"],
    "slimness": ["slim", "slimmest", "thin", "thinnest", "sleek", "thickness"],
    "lightness": ["light", "lightweight", "light weight", "lightest", "weight", "not heavy"],
    "compactness": [
        "compact", "small", "smallest", "smaller", "one hand", "one-hand", "one handed",
        "mini", "pocket", "pocketable"
    ],
    "zoom": ["zoom", "telephoto", "periscope"],
    "multimedia": [
        "multimedia", "movie", "movies", "netflix", "youtube", "video", "videos", "binge",
        "series", "streaming", "entertainment", "big screen", "large screen", "bigger screen", "content"
    ],
    "brand_value": [
        "brand value", "premium brand", "prestigious", "prestige", "status", "brand name",
        "luxury", "reputed brand"
    ],
}

UNIT = r"(k|thousand|lakhs?|lacs?|l)"
NOT_A_PRICE_AFTER = r"(?!\s*(?:mah|w\b|watts?|hz|nits|mp|gb|tb|inch|inches|mm|g\b|grams|%))"
AMOUNT = rf"(?<![a-z0-9.])(\d+(?:\.\d+)?)\s*{UNIT}?(?![a-z0-9]){NOT_A_PRICE_AFTER}"

MIN_CUES = {
    "above", "over", "more than", "at least", "atleast", "minimum", "min",
    "starting", "from", "greater than", "upwards of"
}
MAX_CUES = {
    "under", "below", "within", "up to", "upto", "max", "maximum", "less than", "budget",
    "budget of", "for", "at most", "cheaper than", "till", "until", "not more than", "lower than"
}
POST_MAX_CUES = {"max", "maximum", "or less", "or below", "or under", "budget"}
POST_MIN_CUES = {"or more", "plus", "+", "and above", "or above", "onwards"}


def normalise(sentence):

    text = sentence.lower()
    text = text.replace("₹", " rs ")
    text = re.sub(r"(?<=\d),(?=\d)", "", text)      # 45,000 -> 45000
    text = re.sub(r"\brs\.?\s*", "rs ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def to_rupees(number, unit):

    value = float(number)

    if unit in ("k", "thousand"):
        value *= 1_000
    elif unit and unit.startswith(("lakh", "lac", "l")):
        value *= 100_000

    return int(round(value))


def tokens(text):
    return re.findall(r"[a-z0-9'+]+|,", text)


def cue_before(text, start, cues, max_tokens=3):
    """Distance (in words) to the nearest cue among the last few words before an amount, or None."""

    words = [w for w in tokens(text[:start]) if w not in ("rs", "inr", "rupees")][-max_tokens:]

    for i in range(len(words) - 1, -1, -1):
        candidates = {words[i], " ".join(words[max(0, i - 1):i + 1]), " ".join(words[max(0, i - 2):i + 1])}
        if candidates & cues:
            return len(words) - i

    return None


def cue_after(text, end, cues):
    following = text[end:end + 15].strip()
    return any(following.startswith(cue) for cue in cues)


def extract_prices(text):

    prices = {"min_price": None, "max_price": None, "around_price": None}
    used = []

    # Ranges: "between 15k and 25k", "20 to 30k", "15000-25000"
    range_pattern = rf"(?:between\s+|from\s+)?{AMOUNT}\s*(?:and|to|-|–)\s*(?:rs\s*)?{AMOUNT}"

    for match in re.finditer(range_pattern, text):
        low_number, low_unit, high_number, high_unit = match.groups()
        low = to_rupees(low_number, low_unit or high_unit)    # "20 to 30k": 20 means 20k
        high = to_rupees(high_number, high_unit)
        if low >= 1000 and high >= 1000:
            prices["min_price"], prices["max_price"] = min(low, high), max(low, high)
            used.append(match.span())
            break

    # Approximate: "around 30k", "about 1 lakh"
    around_pattern = rf"\b(?:around|about|approx(?:imately)?|roughly|near|close to|~)\s*(?:rs\s*)?{AMOUNT}"

    for match in re.finditer(around_pattern, text):
        if any(s <= match.start() < e for s, e in used):
            continue
        value = to_rupees(*match.groups())
        if value >= 1000:
            prices["around_price"] = value
            used.append(match.span())
            break

    # Single amounts, with a cue before or after them
    for match in re.finditer(AMOUNT, text):

        if any(s <= match.start() < e for s, e in used):
            continue

        value = to_rupees(*match.groups())
        if value < 1000:
            continue

        min_distance = cue_before(text, match.start(), MIN_CUES)
        max_distance = cue_before(text, match.start(), MAX_CUES)

        if min_distance and (not max_distance or min_distance < max_distance):
            kind = "min_price"
        elif max_distance:
            kind = "max_price"
        elif cue_after(text, match.end(), POST_MIN_CUES):
            kind = "min_price"
        else:
            kind = "max_price"          # a bare amount is read as the budget

        if prices[kind] is None:
            prices[kind] = value
        used.append(match.span())

    return prices, used


NEGATION_WORDS = {"except", "not", "no", "without", "excluding", "exclude", "avoid", "isn't", "isnt", "nor", "hate"}
NEGATION_PHRASES = {
    "anything but", "other than", "apart from", "rather than", "instead of",
    "don't want", "dont want", "do not want", "but not"
}
LIST_WORDS = {"and", "or", "nor", ",", "a", "an", "the", "phone", "phones", "from"}
BRAND_WORDS = {word for alias in BRAND_ALIASES for word in alias.split()} | AMBIGUOUS_BRAND_WORDS


def is_negated(text, start):

    words = tokens(text[:start])

    # Skip back over a list of brands: "except vivo and oppo"
    while words and (words[-1] in LIST_WORDS or words[-1] in BRAND_WORDS):
        words.pop()

    if not words:
        return False

    return words[-1] in NEGATION_WORDS or " ".join(words[-2:]) in NEGATION_PHRASES


TERM_TABLE = (
    [(alias, "brand", brand) for alias, brand in BRAND_ALIASES.items()]
    + [(alias, "processor", processor) for alias, processor in PROCESSOR_ALIASES.items()]
    + [(alias, "display", display) for alias, display in DISPLAY_ALIASES.items()]
    + [(term, "requirement", name) for name, terms in REQUIREMENT_TERMS.items() for term in terms]
    + [(term, "feature", name) for name, terms in FEATURE_TERMS.items() for term in terms]
)


def find_terms(text, used_spans):
    """All known terms in the text. Longest phrases first; matches never overlap; prices are skipped."""

    masked = list(text)
    for s, e in used_spans:
        masked[s:e] = "#" * (e - s)

    found = []

    for phrase, category, value in sorted(TERM_TABLE, key=lambda row: -len(row[0])):
        pattern = rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])"
        for match in re.finditer(pattern, "".join(masked)):
            found.append((match.start(), category, value, phrase))
            masked[match.start():match.end()] = "#" * (match.end() - match.start())

    return sorted(found)


def keyword_extract(sentence):

    text = normalise(sentence)
    prices, used = extract_prices(text)

    found = dict(brands=[], exclude_brands=[], processor_types=[], display_types=[], requirements=[])
    features_in_order = []
    matched_words = []

    for start, category, value, phrase in find_terms(text, used):

        matched_words.append(phrase)

        if category == "brand":
            target = "exclude_brands" if is_negated(text, start) else "brands"
        elif category == "processor":
            target = "processor_types"
        elif category == "display":
            target = "display_types"
        elif category == "requirement":
            target = "requirements"
        else:
            if value not in features_in_order:
                features_in_order.append(value)
            continue

        if value not in found[target]:
            found[target].append(value)

    request = PhoneRequest(
        **prices,
        **found,
        rank_feature=features_in_order[0] if features_in_order else None,
        other_features=features_in_order[1:]
    )

    return request, matched_words


# =========================================================================== language model (Ollama)

DEFAULT_MODEL = "gemma3:1b"


class FeatureGuess(BaseModel):

    rank_feature: Feature | None = Field(
        None, description="The one feature the user cares about most, or null if none"
    )
    evidence: str = Field(
        "", description="The exact words copied from the request that show this feature"
    )


FEATURE_GUIDE = {
    "battery": "battery life, the phone lasting long",
    "performance": "speed, gaming, heavy apps",
    "camera": "taking photos or recording videos",
    "charging": "charging quickly",
    "brightness": "seeing the screen outdoors or in sunlight",
    "refresh_rate": "smooth scrolling and animations",
    "slimness": "a thin phone",
    "lightness": "a light phone",
    "compactness": "a small phone",
    "zoom": "zooming in on far-away things",
    "multimedia": "watching videos and movies",
    "brand_value": "a prestigious or premium brand",
}

FEATURE_PROMPT = "\n".join(
    [
        "You read one smartphone shopping request and decide which ONE feature matters most to the user.",
        "Choose from these features:",
    ]
    + [f"- {name}: {meaning}" for name, meaning in FEATURE_GUIDE.items()]
    + [
        "",
        "Also copy the exact words from the request that show this feature into 'evidence'.",
        "If the request does not point to any of these features, answer null with empty evidence.",
        "",
        "Examples:",
        'Request: my phone heats up and lags in BGMI',
        '{"rank_feature": "performance", "evidence": "lags in BGMI"}',
        'Request: I want to read maps clearly while cycling at noon',
        '{"rank_feature": "brightness", "evidence": "read maps clearly while cycling at noon"}',
        'Request: just a decent phone for my father',
        '{"rank_feature": null, "evidence": ""}',
    ]
)

FILLER_WORDS = (
    BRAND_WORDS
    | {w for cue in MIN_CUES | MAX_CUES | POST_MAX_CUES | POST_MIN_CUES for w in cue.split()}
    | {w for terms in REQUIREMENT_TERMS.values() for term in terms for w in re.findall(r"[a-z']+", term)}
    | {w for alias in list(PROCESSOR_ALIASES) + list(DISPLAY_ALIASES) for w in re.findall(r"[a-z']+", alias)}
    | {
        "a", "an", "the", "i", "me", "my", "we", "you", "it", "is", "are", "am", "be",
        "need", "want", "looking", "show", "some", "any", "phone", "phones", "mobile",
        "smartphone", "new", "one", "with", "and", "or", "of", "in", "on", "to", "that",
        "good", "best", "great", "nice", "decent", "better", "value", "cheap", "affordable",
        "rs", "inr", "rupees", "k", "lakh", "thousand", "please", "get", "buy", "something"
    }
)


def check_evidence(sentence, evidence):
    """Return None if the quoted words support a feature, otherwise the reason they do not."""

    evidence = evidence.strip().lower()

    if not evidence:
        return "no words were quoted"

    if evidence not in sentence.lower():
        return f'the quoted words "{evidence}" are not in the request'

    meaningful = [
        word for word in re.findall(r"[a-z']+", evidence)
        if word not in FILLER_WORDS
    ]

    if not meaningful:
        return f'the quoted words "{evidence}" are only brand, price, requirement, or filler words'

    return None


def call_model(model, messages, schema):

    import ollama   # imported here so the module also works without the Ollama client

    arguments = dict(model=model, messages=messages, format=schema, options={"temperature": 0})

    try:
        response = ollama.chat(**arguments, think=False)
    except (TypeError, ollama.ResponseError):
        # Models without a "thinking" mode, or older clients, do not accept the option
        response = ollama.chat(**arguments)

    return response.message.content


def model_guess_feature(sentence, model=DEFAULT_MODEL):

    answer = call_model(
        model,
        [
            {"role": "system", "content": FEATURE_PROMPT},
            {"role": "user", "content": f"Request: {sentence}"}
        ],
        FeatureGuess.model_json_schema()
    )

    try:
        guess = FeatureGuess.model_validate_json(answer)
    except ValidationError:
        return None, "", "the model's answer was not valid"

    if guess.rank_feature is None:
        return None, "", "the model found no feature"

    problem = check_evidence(sentence, guess.evidence)

    if problem:
        return None, guess.evidence, f"the model suggested {guess.rank_feature}, rejected because {problem}"

    return guess.rank_feature, guess.evidence, "accepted"


# =========================================================================== whole pipeline

def extract_request(sentence, model=DEFAULT_MODEL):
    """Keywords first; the model (if given) is asked only when no feature word was found.

    Pass model=None to use keywords only.
    """

    request, matched_words = keyword_extract(sentence)
    note = "keywords"

    if request.rank_feature is None and model:
        feature, evidence, status = model_guess_feature(sentence, model)

        if feature:
            request = request.model_copy(update={"rank_feature": feature})
            note = f"language model, based on \"{evidence}\""
        else:
            note = status

    return request, note


def describe_request(request, note):
    """The interpretation as (label, value) pairs."""

    rows = [("Budget", budget_text(request))]

    for field_name, label in [
        ("brands", "Brands"),
        ("exclude_brands", "Excluding"),
        ("processor_types", "Processor"),
        ("display_types", "Display"),
        ("requirements", "Requirements"),
    ]:
        values = getattr(request, field_name)
        if values:
            rows.append((label, ", ".join(values)))

    rows.append(("Ranked by", f"{request.rank_feature or 'value score only'} ({note})"))

    if request.other_features:
        rows.append((
            "Also mentioned",
            f"{', '.join(request.other_features)} (only one feature can be ranked at a time for now)"
        ))

    return rows
