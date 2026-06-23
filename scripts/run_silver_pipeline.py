"""
Run the bronze-to-silver ingestion pipeline.

Reads from PostgreSQL, writes Delta tables under data/lakehouse/silver/.
Idempotent — safe to re-run.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from banking_agent.data_pipeline.write_to_delta import write_all_silver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


if __name__ == "__main__":
    write_all_silver()
