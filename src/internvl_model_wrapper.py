class InternVLModelWrapper:
    def __init__(self, *args, **kwargs):
        raise ModuleNotFoundError(
            "InternVLModelWrapper is not included in this trimmed workspace. "
            "Restore the InternVL wrapper before running InternVL experiments."
        )
