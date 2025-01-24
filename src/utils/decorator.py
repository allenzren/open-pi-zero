import torch


def conditional_decorator(dec, condition):
    def decorator(func):
        if not condition:
            # Return the function unchanged, not decorated.
            return func
        return dec(func)

    return decorator


class NoSyncBase:
    def no_sync(self):
        if self.use_ddp:
            # If DDP is used, call the actual `no_sync` method
            return torch.nn.parallel.DistributedDataParallel.no_sync(self)
        else:
            # Otherwise, return the dummy context manager
            class DummyContext:
                def __enter__(self):
                    pass

                def __exit__(self, exc_type, exc_value, traceback):
                    pass

            return DummyContext()


def main_rank_only(func):
    def wrapper(*args, **kwargs):
        if not kwargs.get("main_rank", False):
            return None
        return func(*args, **kwargs)

    return wrapper
