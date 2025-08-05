import toml

class LazyConfig:
    def __init__(self, file_path):
        self.file_path = file_path
        self._config = None

    def _load_config(self):
        if self._config is None:
            self._config = toml.load(self.file_path)
        return self._config
    
    def __getattr__(self, name):
        config = self._load_config()
        if name in config:
            return config[name]
        raise AttributeError(f"'LazyConfig' object has no attribute '{name}'")