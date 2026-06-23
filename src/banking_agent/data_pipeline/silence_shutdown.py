"""
Suppress noisy Windows-specific Spark shutdown stack traces.

Spark on Windows produces harmless 'Failed to delete temp dir' stack traces
during JVM shutdown due to file-handle semantics differences from Linux.
The actual job has already completed successfully at that point — this is
pure cosmetic noise. We redirect stderr at process exit to swallow it.

Production Spark runs on Linux where this pattern works correctly, so this
file is local-development-only hygiene.
"""

import atexit
import os
import sys


def silence_spark_shutdown_on_exit() -> None:
    """Register an atexit handler that redirects stderr to /dev/null equivalent."""
    def _silence() -> None:
        try:
            sys.stderr.flush()
            # On Windows this is the equivalent of /dev/null
            sys.stderr = open(os.devnull, "w")
        except Exception:
            pass

    atexit.register(_silence)