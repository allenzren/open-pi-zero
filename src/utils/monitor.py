import functools
import logging
import time

import torch


def log_allocated_gpu_memory(log=None, stage="loading model", device=0):
    if torch.cuda.is_available():
        allocated_memory = torch.cuda.memory_allocated(device)
        msg = f"Allocated GPU memory after {stage}: {allocated_memory/1024/1024/1024:.2f} GB"
        print(msg) if log is None else log.info(msg)


def log_execution_time(logger=None):
    """Decorator to log the execution time of a function"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            if logger is None:
                print(f"{func.__name__} took {elapsed_time:.2f} seconds to execute.")
            else:
                logger.info(
                    f"{func.__name__} took {elapsed_time:.2f} seconds to execute."
                )
            return result

        return wrapper

    return decorator


class Timer:
    def __init__(self):
        self._start = time.time()

    def __call__(self, reset=True):
        now = time.time()
        diff = now - self._start
        if reset:
            self._start = now
        return diff


# Filter to log only on the main rank
class MainRankFilter(logging.Filter):
    def __init__(self, main_rank):
        super().__init__()
        self.main_rank = main_rank

    def filter(self, record):
        # Only log if this is the main rank
        return self.main_rank
