#!/usr/bin/env python3
"""Generate a synthetic raw ad corpus with DELIBERATELY seeded defects.

Why this exists: without dirty data the validator has nothing to catch and the
report is a wall of zeros. You cannot demonstrate -- or trust -- a data-quality
check you have never seen fire.

Two properties make this useful rather than just noisy:

  1. Every fatal rule in ingestion/rules.py gets representation, at a known rate.
  2. Exactly ONE defect is injected per record, so attribution is unambiguous:
     if a record was seeded as E008, the validator must report E008 for it and
     nothing else.

That second property turns the generator into a TEST ORACLE. It writes
`expected_defects.json` next to the corpus; the validator's report should match
it code for code. If the numbers disagree, either a rule is wrong or the
generator is -- and either way you want to know.

    python scripts/generate_raw_ads.py --count 50000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from bisect import bisect_left
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion import rules  # noqa: E402
from ingestion import schema  # noqa: E402
from ingestion.schema import CATEGORIES  # noqa: E402


# --- Ad copy building blocks ---------------------------------------------

CATALOG: dict[str, dict[str, list[str]]] = {
    "apparel": {
        "brands": ["Northloom", "Verda", "Atlas Thread", "Kestrel", "Maren", "Bly", "Ondra", "Halcyon"],
        "products": ["merino sweater", "linen shirt", "rain shell", "chino trousers", "wool coat",
                     "denim jacket", "thermal base layer", "packable parka", "oxford shirt", "knit cardigan"],
        "hooks": ["built for cold mornings", "cut for everyday wear", "made to outlast the season",
                  "soft on day one", "no-iron finish", "designed in-house"],
    },
    "footwear": {
        "brands": ["Strider", "Panafoot", "Oakstep", "Lumen", "Trailhead", "Cobble & Co", "Rivet", "Solace"],
        "products": ["running shoes", "hiking boots", "leather loafers", "trail runners", "court sneakers",
                     "waterproof boots", "recovery slides", "walking shoes", "cycling shoes", "canvas sneakers"],
        "hooks": ["tested over 500 miles", "broken in from the start", "grip that holds on wet rock",
                  "half sizes in every width", "resoleable, not disposable", "zero break-in period"],
    },
    "electronics": {
        "brands": ["Corvid", "Nimbus Labs", "Byteform", "Aperture One", "Helix", "Quanta", "Ориго", "Pulsewave"],
        "products": ["noise-cancelling headphones", "portable SSD", "mechanical keyboard", "4K monitor",
                     "wireless earbuds", "USB-C hub", "e-reader", "smart display", "action camera", "power bank"],
        "hooks": ["40-hour battery", "plug in and it just works", "firmware updates for five years",
                  "no subscription required", "repairable by design", "ships with the cable you actually need"],
    },
    "home_kitchen": {
        "brands": ["Hearthline", "Copperfield", "Sage & Iron", "Bramble", "Tildon", "Kettleworks", "Oro", "Fennwick"],
        "products": ["cast iron skillet", "espresso machine", "chef's knife", "stand mixer", "air purifier",
                     "cookware set", "electric kettle", "food processor", "dutch oven", "compost bin"],
        "hooks": ["seasoned and ready to cook", "dishwasher safe, genuinely", "parts still sold in ten years",
                  "quiet enough for a studio", "one appliance, not four", "lifetime sharpening included"],
    },
    "beauty": {
        "brands": ["Lumière Co", "Saltwater", "Vellum", "Orris", "Bloomkind", "Aster & Ash", "Petrichor", "Nuvella"],
        "products": ["vitamin C serum", "daily sunscreen", "repair shampoo", "clay mask", "lip balm set",
                     "retinol cream", "cleansing oil", "beard kit", "hand cream", "hair treatment oil"],
        "hooks": ["fragrance free", "dermatologist tested", "no white cast", "works under makeup",
                  "six-ingredient formula", "refills cost half"],
    },
    "fitness": {
        "brands": ["Ironvale", "Repset", "Cadence", "Basecamp", "Vaulter", "Grit House", "Momentum", "Kinetik"],
        "products": ["adjustable dumbbells", "rowing machine", "yoga mat", "resistance band set",
                     "heart rate monitor", "kettlebell pair", "foam roller", "exercise bike", "pull-up bar", "weight vest"],
        "hooks": ["fits under a bed", "quiet enough for apartments", "no monthly membership",
                  "assembles in ten minutes", "used by three national teams", "rated to 300 lbs"],
    },
    "travel": {
        "brands": ["Waypoint", "Longitude", "Carryall", "Meridian", "Roambook", "Portage", "Tessellate", "Farside"],
        "products": ["carry-on suitcase", "travel backpack", "packing cubes", "neck pillow",
                     "universal adapter", "weekender bag", "compression sacks", "toiletry kit", "day pack", "luggage scale"],
        "hooks": ["fits every major airline sizer", "wheels you can replace yourself", "one bag, ten days",
                  "guaranteed for life", "opens flat at security", "weighs under two pounds"],
    },
    "finance": {
        "brands": ["Ledgerly", "Northbank", "Tallyhouse", "Seedplan", "Fairmark", "Abacus Row", "Vaultly", "Onward"],
        "products": ["high-yield savings account", "budgeting app", "tax filing service", "credit monitoring",
                     "index fund portfolio", "business checking", "expense tracker", "retirement planner",
                     "invoice tool", "cash-back card"],
        "hooks": ["no monthly fee", "set up in under five minutes", "your data is never sold",
                  "human support, not a bot", "one flat rate", "no minimum balance"],
    },
    "gaming": {
        "brands": ["Hexbound", "Null Signal", "Ardent Play", "Tactile", "Redshift", "Coldstart", "Overclock", "Mythos"],
        "products": ["wireless controller", "gaming headset", "capture card", "racing wheel",
                     "streaming microphone", "gaming chair", "handheld console", "keypad", "mousepad", "gaming mouse"],
        "hooks": ["sub-1ms latency", "works on every platform", "hot-swappable switches",
                  "no companion app required", "drift-proof sticks", "eight-hour comfort"],
    },
    "pet_supplies": {
        "brands": ["Barkwell", "Thistle & Paw", "Hound House", "Mewtide", "Gentle Leash", "Fernclaw", "Pawline", "Rookery"],
        "products": ["orthopedic dog bed", "grain-free cat food", "automatic feeder", "no-pull harness",
                     "litter box enclosure", "chew toy bundle", "pet water fountain", "grooming kit",
                     "training treats", "travel carrier"],
        "hooks": ["vet formulated", "machine washable cover", "survives a determined chewer",
                  "no plastic smell", "sized for real dogs", "quiet motor"],
    },
    "food_delivery": {
        "brands": ["Sprig Box", "Marketrun", "Thyme Table", "Pantry Line", "Fork & Ferry", "Grainhouse", "Bitewise", "Cellar Door"],
        "products": ["weekly meal kit", "grocery delivery", "ready-made dinners", "coffee subscription",
                     "produce box", "bakery bundle", "family meal plan", "lunch delivery",
                     "specialty pantry crate", "smoothie kit"],
        "hooks": ["on your doorstep by 7am", "skip any week, no charge", "sourced within 100 miles",
                  "twenty minutes to the table", "recyclable packaging", "chef-designed menus"],
    },
    "education": {
        "brands": ["Cortex Academy", "Brightpath", "Studyhall", "Lumen Learn", "Foundry", "Keystone", "Parsed", "Elevate"],
        "products": ["data science course", "language program", "coding bootcamp", "exam prep bundle",
                     "design certificate", "math tutoring", "public speaking workshop", "SQL intensive",
                     "photography class", "writing seminar"],
        "hooks": ["taught by working practitioners", "learn at your own pace", "portfolio project included",
                  "job support for six months", "lifetime access", "small cohorts only"],
    },
}

ADJECTIVES = ["Premium", "Everyday", "Award-winning", "Lightweight", "Handmade", "Best-selling",
              "Professional-grade", "Compact", "All-season", "Refined"]

OFFERS = ["40% Off Today", "Free Shipping This Week", "Buy One Get One", "Save $25 on Your First Order",
          "Limited Spring Release", "New Lower Price", "Try It Free for 30 Days", "Member Pricing Now Live"]

TEMPLATES = [
    "{adj} {product} from {brand} — {offer}",
    "{brand} {product}: {offer}",
    "{offer} on {adj} {product}",
    "Meet the {adj} {brand} {product}",
    "{brand} {product} — {hook}",
    "{adj} {product}, {hook}",
]


# --- Defect mix -----------------------------------------------------------
#
# Fractions of the corpus. One defect per record, so these sum to the overall
# rejection rate (~11.6%) and the validator's report should reproduce them.
DEFECT_RATES: dict[str, float] = {
    rules.E_MALFORMED_JSON: 0.003,
    rules.E_MISSING_FIELD: 0.015,
    rules.E_WRONG_TYPE: 0.008,
    rules.E_EMPTY_TEXT: 0.005,
    rules.E_HEADLINE_TOO_SHORT: 0.010,
    rules.E_TEXT_TOO_LONG: 0.008,
    rules.E_BID_NOT_NUMERIC: 0.006,
    rules.E_BID_OUT_OF_RANGE: 0.020,
    rules.E_UNKNOWN_CATEGORY: 0.010,
    rules.E_DUPLICATE_AD_ID: 0.015,
    rules.E_DUPLICATE_CONTENT: 0.010,
    rules.E_BAD_TIMESTAMP: 0.006,
}

# Cosmetic damage that the cleaner FIXES. These records are still accepted --
# they exist to prove the clean-vs-reject split is real and not theoretical.
NOISE_RATE = 0.15

# Duplicate defects need a valid earlier record to copy from, so they are never
# placed in the first few lines of the file.
MIN_DUP_INDEX = 100


def build_base_record(rng: random.Random, index: int, seen_headlines: set[str]) -> dict:
    category = rng.choice(sorted(CATEGORIES))
    parts = CATALOG[category]

    for _ in range(50):
        headline = rng.choice(TEMPLATES).format(
            adj=rng.choice(ADJECTIVES),
            product=rng.choice(parts["products"]),
            brand=rng.choice(parts["brands"]),
            offer=rng.choice(OFFERS),
            hook=rng.choice(parts["hooks"]).capitalize(),
        )
        if len(headline) <= 120 and headline not in seen_headlines:
            break
    else:  # pragma: no cover - astronomically unlikely with this catalog size
        headline = f"{rng.choice(parts['products']).title()} — offer #{index}"
    seen_headlines.add(headline)

    description = (
        f"{rng.choice(parts['hooks']).capitalize()}. "
        f"{rng.choice(parts['brands'])} {rng.choice(parts['products'])} "
        f"for people who {rng.choice(['shop carefully', 'hate replacing things', 'read the spec sheet', 'want it to just work'])}. "
        f"{rng.choice(OFFERS)}."
    )

    # Bids cluster low with a long right tail, like real CPMs.
    bid = round(min(rng.lognormvariate(0.9, 0.6), 45.0), 2)

    age_days = rng.random() ** 2 * 180  # skewed towards recent
    created = datetime.now(timezone.utc) - timedelta(days=age_days)

    return {
        "ad_id": f"ad_{index:07d}",
        "headline": headline,
        "description": description,
        "category": category,
        "advertiser_id": f"adv_{rng.randint(1, 400):04d}",
        "bid_cpm_usd": bid,
        "created_at": created.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# --- Defect injectors -----------------------------------------------------
# Each mutates one record in place so that exactly one rule fires.


def _repeat_past(text: str, limit: int) -> str:
    """Repeat `text` until it is comfortably past `limit` characters.

    A fixed multiplier is not good enough, and the corpus proved it: the
    shortest generated description repeated 12x lands at ~912 characters,
    under the 1000 limit, so one seeded defect was silently accepted and the
    oracle came up one short. Size the repetition from the limit instead.
    """
    unit = text + " "
    return unit * (limit // len(unit) + 2)


def inject(code: str, record: dict, rng: random.Random, source: dict | None) -> None:
    if code == rules.E_MALFORMED_JSON:
        # Written verbatim instead of being serialised -- a truncated line, the
        # classic result of a writer process dying mid-flush.
        record["__raw_line__"] = json.dumps(record)[: rng.randint(20, 60)]

    elif code == rules.E_MISSING_FIELD:
        field = rng.choice(["headline", "category", "bid_cpm_usd", "advertiser_id", "created_at"])
        if rng.random() < 0.5:
            record.pop(field)
        else:
            record[field] = None  # present but null is just as unusable

    elif code == rules.E_WRONG_TYPE:
        field, value = rng.choice(
            [
                ("headline", ["Premium", "running", "shoes"]),
                ("category", 7),
                ("advertiser_id", {"id": 42}),
                ("created_at", 1789000000),
            ]
        )
        record[field] = value

    elif code == rules.E_EMPTY_TEXT:
        record["headline"] = rng.choice(["   ", "\t\n ", "<b></b>", "&nbsp;"])

    elif code == rules.E_HEADLINE_TOO_SHORT:
        record["headline"] = rng.choice(["Sale!", "Buy now", "50% off", "New", "Shop", "Deal"])

    elif code == rules.E_TEXT_TOO_LONG:
        if rng.random() < 0.5:
            record["headline"] = _repeat_past(record["headline"], schema.HEADLINE_MAX_LEN)
        else:
            record["description"] = _repeat_past(record["description"], schema.DESCRIPTION_MAX_LEN)

    elif code == rules.E_BID_NOT_NUMERIC:
        record["bid_cpm_usd"] = rng.choice(["N/A", "not set", "TBD", "--", "auto", ["4.25"]])

    elif code == rules.E_BID_OUT_OF_RANGE:
        record["bid_cpm_usd"] = rng.choice(
            [
                -round(rng.uniform(0.5, 9.0), 2),          # negative
                0.0,                                        # zero
                round(rng.uniform(100, 900), 0),            # cents sent as dollars
                round(rng.uniform(60, 5000), 2),            # fat-fingered
            ]
        )

    elif code == rules.E_UNKNOWN_CATEGORY:
        record["category"] = rng.choice(
            ["misc", "other", "uncategorized", "general", "sports", "toys", "automotive", ""]
        )

    elif code == rules.E_DUPLICATE_AD_ID:
        assert source is not None
        record["ad_id"] = source["ad_id"]  # own copy keeps its distinct text

    elif code == rules.E_DUPLICATE_CONTENT:
        assert source is not None
        record["headline"] = source["headline"]
        record["description"] = source["description"]

    elif code == rules.E_BAD_TIMESTAMP:
        record["created_at"] = rng.choice(
            ["14/03/2026", "yesterday", "2026-13-45T00:00:00Z", "March 14 2026", ""]
        )

    else:  # pragma: no cover
        raise ValueError(f"no injector for {code}")


def apply_noise(record: dict, rng: random.Random) -> None:
    """Damage that the cleaner repairs. The record must still be ACCEPTED."""
    choice = rng.randrange(5)
    if choice == 0:
        record["headline"] = f"<b>{record['headline']}</b>"
    elif choice == 1:
        record["headline"] = f"   {record['headline']}  \n\t "
    elif choice == 2:
        record["headline"] = record["headline"].replace(" ", "​ ", 1)
    elif choice == 3:
        record["bid_cpm_usd"] = f"${record['bid_cpm_usd']:.2f}"
    else:
        record.pop("description")


def generate(count: int, seed: int, out_path: Path) -> dict:
    rng = random.Random(seed)
    seen_headlines: set[str] = set()
    records = [build_base_record(rng, i, seen_headlines) for i in range(count)]

    # --- plan which records get which defect ------------------------------
    slots = list(range(count))
    rng.shuffle(slots)

    dup_codes = (rules.E_DUPLICATE_AD_ID, rules.E_DUPLICATE_CONTENT)
    assignment: dict[int, str] = {}

    # Duplicates are placed first because they are the only defect with a
    # position constraint (they need an earlier clean record to copy).
    for code in dup_codes:
        needed = int(count * DEFECT_RATES[code])
        taken = [i for i in slots if i >= MIN_DUP_INDEX and i not in assignment][:needed]
        for i in taken:
            assignment[i] = code

    for code, rate in DEFECT_RATES.items():
        if code in dup_codes:
            continue
        needed = int(count * rate)
        taken = [i for i in slots if i not in assignment][:needed]
        for i in taken:
            assignment[i] = code

    # Pristine records are the safe source for duplicate copying: they are
    # neither defective nor noised, so their text survives cleaning unchanged.
    noise_pool = [i for i in slots if i not in assignment]
    noise_count = int(count * NOISE_RATE)
    noised = set(noise_pool[:noise_count])
    pristine = sorted(i for i in noise_pool[noise_count:])

    # --- apply ------------------------------------------------------------
    for index, code in sorted(assignment.items()):
        source = None
        if code in dup_codes:
            cut = bisect_left(pristine, index)
            if cut == 0:
                continue  # no earlier pristine record; leave this one clean
            source = records[pristine[rng.randrange(cut)]]
        inject(code, records[index], rng, source)

    for index in noised:
        apply_noise(records[index], rng)

    # --- write ------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            raw_line = record.pop("__raw_line__", None)
            handle.write((raw_line if raw_line is not None else json.dumps(record, ensure_ascii=False)) + "\n")

    expected: dict[str, int] = {}
    for code in assignment.values():
        expected[code] = expected.get(code, 0) + 1

    return {
        "seed": seed,
        "count": count,
        "expected_rejections_by_code": dict(sorted(expected.items())),
        "expected_rejected_total": sum(expected.values()),
        "expected_accepted_total": count - sum(expected.values()),
        "noised_records": len(noised),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a synthetic raw ad corpus.")
    parser.add_argument("--count", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data/raw/ads_raw.jsonl"))
    args = parser.parse_args()

    summary = generate(args.count, args.seed, args.out)

    oracle_path = args.out.parent / "expected_defects.json"
    oracle_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {summary['count']:,} raw ads -> {args.out}")
    print(f"seeded {summary['expected_rejected_total']:,} defective records "
          f"({summary['expected_rejected_total'] / summary['count']:.2%}) "
          f"and {summary['noised_records']:,} cleanable-but-valid records")
    print(f"oracle -> {oracle_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
