# This is a singleton class to maintain list of models
class ModelManager:
    _instance = None
    _models = None
    
    def __new__(cls, *args, **kwargs):
        # Ensure that only one instance of ModelManager is created
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls, *args, **kwargs)
            cls._instance._models = {}  # Initialize the list of models

        return cls._instance
    
    def get_all_models(self):
        return self._models

    def add_model(self, model):
        """Add a model to the list."""
        if (self._models is None):
            self._models = {}
        self._models.update(model)


    def get_model_by_name(self, model_name: str):
        """Get the list of models."""
        return self._models
    
    def clear_models(self):
        """Clear the list of models."""
        self._models.clear()
        self._models = {}

    def __repr__(self):
        return f"ModelManager(models={self._models})"

