"""Exceptions raised when execution policy cannot be applied."""


class PolicyError(ValueError):
    """Raised when a profile document or task grant is invalid."""


class PolicyWidening(PolicyError):
    """Raised when a delegated request exceeds its parent authority."""


class PolicyDenied(PolicyError):
    """Raised when code directly invokes a capability hidden by policy."""

    def __init__(self, capability: str, message: str | None = None) -> None:
        self.capability = capability
        super().__init__(message or f"capability {capability!r} is not granted by the active policy")
