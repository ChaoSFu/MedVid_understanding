"""Frozen model backends; mock inference is always marked synthetic."""


def make_backend(config):
    from .mock import MockBackend
    from .local_hf import LocalHFBackend
    from .openai_compatible import OpenAICompatibleBackend
    if config["kind"] == "mock":
        return MockBackend(config)
    if config["kind"] == "openai_compatible":
        return OpenAICompatibleBackend(config)
    if config["kind"] == "local_hf":
        return LocalHFBackend(config)
    raise ValueError("Unsupported backend kind")
