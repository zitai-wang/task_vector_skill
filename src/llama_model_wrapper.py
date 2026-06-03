class LlamaModelWrapper:
    def __init__(self, *args, **kwargs):
        raise ModuleNotFoundError(
            "LlamaModelWrapper is not included in this trimmed workspace. "
            "Use the Qwen text pipeline or restore the llama wrapper before running llama experiments."
        )
