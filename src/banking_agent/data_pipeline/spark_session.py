"""
SparkSession factory configured for Delta Lake operations.

The session is constructed once per process and reused. Delta Lake requires
specific configuration to register its catalog extensions and SQL syntax —
this module captures that configuration in one place so the rest of the
pipeline code can ignore it.
"""

import logging
from functools import lru_cache

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_spark() -> SparkSession:
    """
    Construct or return the singleton SparkSession.

    Local mode runs executors as threads in the driver process. Production
    deployments configure cluster-mode execution against YARN, Kubernetes,
    or Databricks — the application code does not change, only the master URL.

    Delta Lake integration is registered through configure_spark_with_delta_pip,
    which downloads the Delta JARs on first run and adds them to the classpath.
    Additional JVM dependencies — the PostgreSQL JDBC driver — are passed
    through extra_packages so they merge with Delta's configuration rather
    than being overwritten by it.
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
        .config("spark.log.level", "WARN")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.adaptive.enabled", "true")
    )

    spark = configure_spark_with_delta_pip(
        builder,
        extra_packages=["org.postgresql:postgresql:42.7.4"],
    ).getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    logger.info("spark_session_ready version=%s", spark.version)
    return spark