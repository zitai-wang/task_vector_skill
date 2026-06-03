class QwenVLModelWrapper:
    def __init__(self, *args, **kwargs):
        raise ModuleNotFoundError(
            "QwenVLModelWrapper is not included in this trimmed workspace. "
            "Use the Qwen text pipeline or restore the VL wrapper before running Qwen-VL experiments."
        )
