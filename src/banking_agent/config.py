"""
Configuration loading for the banking agent.

Reads environment variables and validates them through Pydantic. The application
never reads os.environ directly outside this module — all configuration access
goes through the Config singleton. This is the production pattern: configuration
is loaded once at startup, validated once, and accessed through a typed object
for the lifetime of the process.
"""

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env file before any environment variable reads.
# In production (Lambda, ECS), environment variables are injected by the runtime
# and this is a no-op. In local development, this populates os.environ from .env.
_PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


class Config(BaseSettings):
    """
    Typed application configuration.

    Pydantic Settings reads environment variables, validates them against the
    declared types, and rejects malformed values at startup. A configuration
    error at startup is preferable to a configuration error at request time —
    the process fails fast instead of failing under load.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # AWS configuration
    aws_region: str = Field(default="us-east-1", description="AWS region for all service calls")
    aws_profile: str = Field(default="default", description="AWS CLI profile to use")

    # Bedrock model configuration
    bedrock_model_id: str = Field(
        default="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        description=(
            "Bedrock model identifier. Inference profile IDs are preferred "
            "over base model IDs."
        ),
    )
    bedrock_temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    bedrock_max_tokens: int = Field(default=1024, gt=0, le=8192)

    # Database connection strings
    postgres_dsn: str = Field(
        default="postgresql://agent:agent_dev_password@localhost:5432/banking",
        description="PostgreSQL connection string in libpq URI format.",
    )
    mongodb_uri: str = Field(
        default="mongodb://agent:agent_dev_password@localhost:27017/banking?authSource=admin",
        description="MongoDB connection URI.",
    )
    redis_url: str = Field(
        default="redis://:agent_dev_password@localhost:6379/0",
        description="Redis connection URL.",
    )

    # Cache TTLs (seconds)
    cache_ttl_account: int = Field(default=30, gt=0)
    cache_ttl_customer: int = Field(default=300, gt=0)

    # Connection pool sizing
    postgres_pool_min: int = Field(default=2, ge=1)
    postgres_pool_max: int = Field(default=10, ge=1)

    # Agent configuration
    agent_max_turns: int = Field(
        default=10,
        gt=0,
        le=50,
        description=(
            "Hard limit on agent loop iterations. Prevents runaway costs "
            "from infinite tool calls."
        ),
    )

    # Logging
    log_level: str = Field(default="INFO", pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")


@lru_cache(maxsize=1)
def get_config() -> Config:
    """
    Returns the singleton Config instance.

    The lru_cache decorator ensures the Config is constructed once per process.
    First call validates environment variables and constructs the object;
    subsequent calls return the cached instance. This is the production
    pattern for application configuration.
    """
    return Config()
