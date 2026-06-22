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