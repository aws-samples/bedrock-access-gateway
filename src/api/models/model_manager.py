# Original Credit: GitHub user dhapola 


class ModelManager:

    @property
    def model_keys(self):
        return list(self.get_all_models().keys())

    def __init__(self, *args, **kwargs):
        self._models = {}

    def get_all_models(self):
        return self._models

    def add_model(self, model):
        """Add a model to the list."""
        self._models.update(model)

    def clear_models(self):
        """Clear the list of models."""
        self._models.clear()
        self._models = {}

    def __repr__(self):
        return f"ModelManager(models={self._models})"