"""
Generate synthetic customer change events for SCD Type 2 demonstration.

In production, these events would arrive from a change-data-capture (CDC)
stream from the operational database. For the portfolio project we synthesize
a believable stream: 30% of customers experience an address change at some
point in the past 18 months, 20% experience a risk rating change.

Output: a CSV file under data/changes/ that we will load into the Delta
silver layer and merge into the customer history table.
"""

import csv
import logging
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import psycopg
from faker import Faker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

fake = Faker()
random.seed(42)  # reproducible

US_STATES = ["CA", "NY", "TX", "FL", "PA", "IL", "OH", "GA", "NC", "MI"]
RISK_RATINGS = ["LOW", "MEDIUM", "HIGH"]

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "changes"
OUTPUT_FILE = OUTPUT_DIR / "customer_changes.csv"

POSTGRES_URL = "postgresql://agent:agent_dev_password@localhost:5432/banking"


def fetch_current_customers() -> list[dict]:
    """Read the current customer state from the operational database."""
    with psycopg.connect(POSTGRES_URL) as conn:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute("""
                SELECT customer_id, address_line, state_code, risk_rating
                FROM customers
                ORDER BY customer_id
            """)
            return cur.fetchall()


def generate_changes(customers: list[dict]) -> list[dict]:
    """
    For each customer, generate 0, 1, or 2 change events over the past 18 months.

    The change events are the deltas — the new values that supersede the old.
    Each event has a change_timestamp and the affected attribute(s).
    """
    changes = []
    now = datetime.now()

    for customer in customers:
        # 30% chance of address change
        if random.random() < 0.30:
            change_time = now - timedelta(days=random.randint(30, 540))
            changes.append({
                "customer_id": customer["customer_id"],
                "change_timestamp": change_time.isoformat(),
                "address_line": fake.street_address(),
                "state_code": random.choice(US_STATES),
                "risk_rating": customer["risk_rating"],  # unchanged
                "change_type": "ADDRESS",
            })

        # 20% chance of risk rating change (independent of address change)
        if random.random() < 0.20:
            change_time = now - timedelta(days=random.randint(30, 540))
            new_rating = random.choice([r for r in RISK_RATINGS if r != customer["risk_rating"]])
            changes.append({
                "customer_id": customer["customer_id"],
                "change_timestamp": change_time.isoformat(),
                "address_line": customer["address_line"],  # unchanged
                "state_code": customer["state_code"],  # unchanged
                "risk_rating": new_rating,
                "change_type": "RISK_RATING",
            })

    # Sort by timestamp — order matters for SCD Type 2 because each change
    # references the state at the time of the change
    changes.sort(key=lambda c: c["change_timestamp"])
    return changes


def write_changes(changes: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["customer_id", "change_timestamp", "address_line",
                        "state_code", "risk_rating", "change_type"],
        )
        writer.writeheader()
        writer.writerows(changes)
    logger.info("wrote_changes count=%d path=%s", len(changes), OUTPUT_FILE)


def main() -> None:
    customers = fetch_current_customers()
    logger.info("loaded_customers count=%d", len(customers))
    changes = generate_changes(customers)
    logger.info("generated_changes count=%d address=%d risk=%d",
                len(changes),
                sum(1 for c in changes if c["change_type"] == "ADDRESS"),
                sum(1 for c in changes if c["change_type"] == "RISK_RATING"))
    write_changes(changes)


if __name__ == "__main__":
    main()
