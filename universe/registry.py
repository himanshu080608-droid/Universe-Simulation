class PresetRegistry:
    _presets = {}

    @classmethod
    def register(cls, preset_def):
        cls._presets[preset_def.name] = preset_def

    @classmethod
    def get(cls, name):
        if name not in cls._presets:
            raise ValueError(f"Preset '{name}' not found in registry.")
        return cls._presets[name]

    @classmethod
    def list_presets(cls):
        return list(cls._presets.keys())
