"""
SparkSession factory configured for Delta Lake operations.

The session is constructed once per process and reused. Delta Lake requires
specific configuration to register its catalog extensions and SQL syntax —
this module captures that configuration in one place so the rest of the
pipeline code can ignore it.

JVM-side logging is suppressed to ERROR level so only meaningful output
reaches the terminal. Production Databricks deployments do the same through
cluster log configuration.
"""

import logging
import os
import sys
from functools import lru_cache

# Pin the Python interpreter for both driver and executor workers.
# Production discipline: explicit Python pinning prevents version drift
# between driver and executors in any deployment.
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Suppress py4j's logger before pyspark is even imported — it logs every
# JVM call at INFO level by default, which floods the terminal during
# Spark operations.
logging.getLogger("py4j").setLevel(logging.WARNING)
logging.getLogger("py4j.java_gateway").setLevel(logging.WARNING)
logging.getLogger("py4j.clientserver").setLevel(logging.WARNING)

from delta import configure_spark_with_delta_pip  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_spark() -> SparkSession:
    """
    Construct or return the singleton SparkSession.

    Local mode runs executors as threads in the driver process. Production
    deployments configure cluster-mode execution against YARN, Kubernetes,
    or Databricks — the application code does not change, only the master URL.
    """
    builder = (
        SparkSession.builder
        .appName("pnc-banking-data-pipeline")
        .master("local[*]")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.ui.showConsoleProgress", "false")
        # Suppress JVM-side INFO/WARN logging — leaves only errors visible
        .config("spark.log.level", "ERROR")

        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.adaptive.enabled", "true")
    )

    spark = configure_spark_with_delta_pip(
        builder,
        extra_packages=["org.postgresql:postgresql:42.7.4"],
    ).getOrCreate()

    # Set the SparkContext log level to ERROR — covers loggers that the
    # builder config does not catch (executor logs, shutdown hooks, GC).
    spark.sparkContext.setLogLevel("ERROR")

    logger.info("spark_session_ready version=%s", spark.version)
    return spark
