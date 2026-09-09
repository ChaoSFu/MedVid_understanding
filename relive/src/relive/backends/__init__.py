"""Frozen model backends; mock inference is always marked synthetic."""


def make_backend(config):
    from .mock import MockBackend
    from .openai_compatible import OpenAICompatibleBackend
    if config["kind"] == "mock":
        return MockBackend(config)
    if config["kind"] == "openai_compatible":
        return OpenAICompatibleBackend(config)
    raise ValueError("Unsupported backend kind")
