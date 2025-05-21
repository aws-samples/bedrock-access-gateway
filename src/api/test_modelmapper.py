import unittest
from unittest.mock import patch, mock_open
from api.modelmapper import get_model, load_model_map

@patch("api.modelmapper._model_map", {
    "provider1": {
        "model1": "mapped_model1",
        "model2": "mapped_model2",
        "model3:latest": "mapped_model3"
    }
})
class TestModelMapper(unittest.TestCase):
    def test_get_model_with_existing_model(self):
        result = get_model("provider1", "model1")
        self.assertEqual(result, "mapped_model1")

    @patch("api.modelmapper.FALLBACK_MODEL", "fallback_model")
    def test_get_model_with_non_existing_model(self):
        result = get_model("provider1", "non_existing_model")
        self.assertEqual(result, "fallback_model")

    @patch("api.modelmapper.FALLBACK_MODEL", None)
    def test_get_model_with_no_fallback(self):
        result = get_model("provider1", "non_existing_model")
        self.assertEqual(result, "non_existing_model")

    @patch("api.modelmapper._model_map", {
        "provider1": {
            "model1": "mapped_model1"
        }
    })

    @patch("api.modelmapper.FALLBACK_MODEL", "fallback_model")
    def test_get_model_with_case_insensitivity(self):
        result = get_model("PROVIDER1", "MODEL1")
        self.assertEqual(result, "mapped_model1")

    def test_get_model_with_latest(self):
        result = get_model("provider1", "model3:latest")
        self.assertEqual(result, "mapped_model3")

    @patch("builtins.open", new_callable=mock_open, read_data='{"provider1": {"model1": "mapped_model1"}}')
    @patch("os.path.join", return_value="/mocked/path/modelmap.json")
    @patch("os.path.dirname", return_value="/mocked/path")
    @patch("os.path.abspath", return_value="/mocked/path/modelmapper.py")
    def test_load_model_map(self, mock_abspath, mock_dirname, mock_join, mock_open_file):
        import api.modelmapper as modelmapper  # <- directly access the module
        modelmapper._model_map = None  # Reset the actual global used by load_model_map
        modelmapper.load_model_map()
        self.assertEqual(
            modelmapper._model_map,
            {"provider1": {"model1": "mapped_model1"}}
        )
        
if __name__ == "__main__":
    unittest.main()
