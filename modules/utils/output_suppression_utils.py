import os
import sys
import contextlib
from transformers import logging as hf_logging

def silence_transformers():
    """Suppress transformers library logging."""
    hf_logging.set_verbosity_error()

@contextlib.contextmanager
def suppress_output():
    """
    Context manager to suppress stdout and stderr output (e.g. from noisy libraries like NeMo).
    Usage:
        with suppress_output():
            # code that prints a lot
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
