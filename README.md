# PNC Banking Agent

A production-pattern customer service agent for retail banking, built on LangGraph and Amazon Bedrock.

## Status

Active development. Not for production use.

## Architecture

LangGraph state machine orchestrating tool-calling against banking domain APIs, with Claude Sonnet 4.5 on Amazon Bedrock as the underlying LLM. Retrieval-augmented generation over public banking policy documents (FDIC, CFPB). Deployed on AWS as a Lambda-fronted API Gateway endpoint with DynamoDB conversation state and OpenSearch vector storage.

## Local Development

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Copy and configure environment variables
copy .env.example .env

# Run the agent locally
python scripts/run_local.py
```

## License

MIT

## Slowly Changing Dimensions Type 2

Customer profile history is maintained as an SCD Type 2 Delta table.
The table tracks every historical version of each customer's address,
state of residence, and internal risk rating, with effective-date ranges
and a current-row flag. This enables point-in-time queries — "what was
this customer's state of residence when this transaction was approved"
— which is foundational for banking compliance and temporal analytics.

The SCD Type 2 merge logic is implemented in PySpark SQL and executed
on Databricks. The local development environment writes the silver
layer and generates synthetic change events; the merge itself runs on
Databricks because the production target for this pattern is Databricks
on Linux. The merge is atomic — the close-out of the old version and
the insert of the new version happen in a single transaction.

### Pattern demonstrated
- Surrogate key + natural key separation
- Effective-from / effective-to date ranges
- Current-row flag (denormalized for query performance)
- Atomic close-out-and-insert via Delta MERGE INTO
- Point-in-time reconstruction via standard SCD2 predicate

### Known simplification
The demo applies one change per customer per merge run. Production
SCD2 pipelines handling multi-change-per-entity batches require
either upstream deduplication or window-function-based change
sequencing before merge. The current implementation uses the dedupe
approach. Future enhancement: implement change sequencing for full
multi-change batches.

### Screenshots
- Merge execution: `docs/scd2_merge_success.png`
- Customer version history: `docs/scd2_customer_history.png`
- Point-in-time query: `docs/scd2_point_in_time.png`