class DummyBackend:
    """
    Dummy backend for tests and examples.
    """

    @property
    def name(self) -> str:
        return "dummy"

    def complete(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
    ) -> str:
        return '{"label": "dummy", "rationale": "This is a dummy response."}'
