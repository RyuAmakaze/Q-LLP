class DummyTqdm:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable
    def __iter__(self):
        return iter(self.iterable) if self.iterable is not None else iter([])
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        pass
    def update(self, *args, **kwargs):
        pass
    def close(self):
        pass

def tqdm(iterable=None, *args, **kwargs):
    return DummyTqdm(iterable)
