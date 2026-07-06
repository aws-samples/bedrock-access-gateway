import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from api.schema import ChatRequest, JsonSchema, ResponseFormat

# ---------------------------------------------------------------------------
# Schema parsing tests
# ---------------------------------------------------------------------------


class TestResponseFormatSchema:
    def _request(self, **overrides):
        base = {"messages": [{"role": "user", "content": "hi"}], "model": "test"}
        base.update(overrides)
        return ChatRequest.model_validate(base)

    def test_no_response_format(self):
        req = self._request()
        assert req.response_format is None

    def test_text_type(self):
        req = self._request(response_format={"type": "text"})
        assert req.response_format.type == "text"

    def test_json_object_type(self):
        req = self._request(response_format={"type": "json_object"})
        assert req.response_format.type == "json_object"
        assert req.response_format.json_schema is None

    def test_json_schema_type(self):
        req = self._request(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "person",
                    "description": "A person",
                    "schema": {"type": "object", "properties": {"name": {"type": "string"}}},
                    "strict": True,
                },
            }
        )
        assert req.response_format.type == "json_schema"
        js = req.response_format.json_schema
        assert js.name == "person"
        assert js.description == "A person"
        assert js.schema_["type"] == "object"
        assert js.strict is True

    def test_json_schema_minimal(self):
        req = self._request(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "data",
                    "schema": {"type": "object"},
                },
            }
        )
        js = req.response_format.json_schema
        assert js.description is None
        assert js.strict is None

    def test_invalid_type_rejected(self):
        with pytest.raises(ValidationError):
            self._request(response_format={"type": "xml"})


# ---------------------------------------------------------------------------
# _ensure_additional_properties tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bedrock_model_class():
    """Import BedrockModel with boto3 calls mocked out."""
    with patch(
        "boto3.client",
        return_value=MagicMock(
            list_inference_profiles=MagicMock(return_value={"inferenceProfileSummaries": []}),
            list_foundation_models=MagicMock(return_value={"modelSummaries": []}),
        ),
    ):
        from api.models.bedrock import BedrockModel
    return BedrockModel


class TestEnsureAdditionalProperties:
    def test_simple_object(self, bedrock_model_class):
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        result = bedrock_model_class._ensure_additional_properties(schema)
        assert result["additionalProperties"] is False

    def test_nested_object(self, bedrock_model_class):
        schema = {
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "properties": {"b": {"type": "integer"}},
                },
            },
        }
        result = bedrock_model_class._ensure_additional_properties(schema)
        assert result["additionalProperties"] is False
        assert result["properties"]["nested"]["additionalProperties"] is False

    def test_array_items(self, bedrock_model_class):
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"x": {"type": "number"}},
            },
        }
        result = bedrock_model_class._ensure_additional_properties(schema)
        assert result["items"]["additionalProperties"] is False

    def test_preserves_explicit_value(self, bedrock_model_class):
        schema = {"type": "object", "additionalProperties": True}
        result = bedrock_model_class._ensure_additional_properties(schema)
        assert result["additionalProperties"] is True

    def test_defs(self, bedrock_model_class):
        schema = {
            "type": "object",
            "$defs": {
                "Address": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
            "properties": {"addr": {"$ref": "#/$defs/Address"}},
        }
        result = bedrock_model_class._ensure_additional_properties(schema)
        assert result["$defs"]["Address"]["additionalProperties"] is False

    def test_anyof(self, bedrock_model_class):
        schema = {
            "anyOf": [
                {"type": "object", "properties": {"a": {"type": "string"}}},
                {"type": "string"},
            ]
        }
        result = bedrock_model_class._ensure_additional_properties(schema)
        assert result["anyOf"][0]["additionalProperties"] is False

    def test_non_dict_passthrough(self, bedrock_model_class):
        assert bedrock_model_class._ensure_additional_properties("not a dict") == "not a dict"

    def test_does_not_mutate_input(self, bedrock_model_class):
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        bedrock_model_class._ensure_additional_properties(schema)
        assert "additionalProperties" not in schema


# ---------------------------------------------------------------------------
# _parse_request response_format mapping tests
# ---------------------------------------------------------------------------


class TestParseRequestResponseFormat:
    @pytest.fixture(autouse=True)
    def setup(self, bedrock_model_class):
        self.model = bedrock_model_class.__new__(bedrock_model_class)

    def _request(self, **overrides):
        base = {
            "messages": [{"role": "user", "content": "hello"}],
            "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        }
        base.update(overrides)
        return ChatRequest.model_validate(base)

    def test_no_response_format_no_output_config(self):
        req = self._request()
        args = self.model._parse_request(req)
        assert "outputConfig" not in args

    def test_text_type_no_output_config(self):
        req = self._request(response_format={"type": "text"})
        args = self.model._parse_request(req)
        assert "outputConfig" not in args

    def test_json_schema_produces_output_config(self):
        req = self._request(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "result",
                    "description": "Test result",
                    "schema": {
                        "type": "object",
                        "properties": {"score": {"type": "integer"}},
                        "required": ["score"],
                    },
                },
            }
        )
        args = self.model._parse_request(req)
        oc = args["outputConfig"]
        assert oc["textFormat"]["type"] == "json_schema"
        js_def = oc["textFormat"]["structure"]["jsonSchema"]
        assert js_def["name"] == "result"
        assert js_def["description"] == "Test result"
        parsed_schema = json.loads(js_def["schema"])
        assert parsed_schema["additionalProperties"] is False
        assert parsed_schema["properties"]["score"]["type"] == "integer"

    def test_json_schema_schema_is_string(self):
        req = self._request(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "x",
                    "schema": {"type": "object"},
                },
            }
        )
        args = self.model._parse_request(req)
        schema_val = args["outputConfig"]["textFormat"]["structure"]["jsonSchema"]["schema"]
        assert isinstance(schema_val, str)

    def test_json_schema_no_description_omitted(self):
        req = self._request(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "x",
                    "schema": {"type": "object"},
                },
            }
        )
        args = self.model._parse_request(req)
        js_def = args["outputConfig"]["textFormat"]["structure"]["jsonSchema"]
        assert "description" not in js_def

    def test_json_object_appends_system_prompt(self):
        req = self._request(response_format={"type": "json_object"})
        args = self.model._parse_request(req)
        assert "outputConfig" not in args
        system_texts = [s["text"] for s in args["system"]]
        assert any("JSON" in t for t in system_texts)

    def test_json_object_preserves_existing_system(self):
        req = self._request(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hello"},
            ],
            response_format={"type": "json_object"},
        )
        args = self.model._parse_request(req)
        system_texts = [s["text"] for s in args["system"]]
        assert any("helpful" in t for t in system_texts)
        assert any("JSON" in t for t in system_texts)
