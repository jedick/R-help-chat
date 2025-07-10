from contextlib import contextmanager
import os
import sys

@contextmanager
def SuppressStderr():
    """
    Context for suppressing stderr messages from Chroma:
    # Failed to send telemetry event ClientStartEvent: capture() takes 1 positional argument but 3 were given
    # Failed to send telemetry event ClientCreateCollectionEvent: capture() takes 1 positional argument but 3 were given
    """
    try:
        # Save the original stderr
        original_stderr = sys.stderr
        sys.stderr = open(os.devnull, "w")  # Redirect to null device
        yield  # Code inside the `with` block executes here
    finally:
        # Restore stderr
        sys.stderr = original_stderr


