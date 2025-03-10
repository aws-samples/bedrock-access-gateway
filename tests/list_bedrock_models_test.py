from typing import Literal

from src.api.models.bedrock import list_bedrock_models, BedrockClientInterface

def test_default_model():
    client = FakeBedrockClient(
        inference_profile("p1-id", "p1", "SYSTEM_DEFINED"),
        inference_profile("p2-id", "p2", "APPLICATION"),
        inference_profile("p3-id", "p3", "SYSTEM_DEFINED"),
    )

    models = list_bedrock_models(client)

    assert models == {
        "anthropic.claude-3-sonnet-20240229-v1:0": {
            "modalities": ["TEXT", "IMAGE"]
        }
    }

def test_one_model():
    client = FakeBedrockClient(
        model("model-id", "model-name", stream_supported=True, input_modalities=["TEXT", "IMAGE"])
    )

    models = list_bedrock_models(client)

    assert models == {
        "model-id": {
            "modalities": ["TEXT", "IMAGE"]
        }
    }

def test_two_models():
    client = FakeBedrockClient(
        model("model-id-1", "model-name-1", stream_supported=True, input_modalities=["TEXT", "IMAGE"]),
        model("model-id-2", "model-name-2", stream_supported=True, input_modalities=["IMAGE"])
    )

    models = list_bedrock_models(client)

    assert models == {
        "model-id-1": {
            "modalities": ["TEXT", "IMAGE"]
        },
        "model-id-2": {
            "modalities": ["IMAGE"]
        }
    }

def test_filter_models():
    client = FakeBedrockClient(
        model("model-id", "model-name-1", stream_supported=True, input_modalities=["TEXT"], status="LEGACY"),
        model("model-id-no-stream", "model-name-2", stream_supported=False, input_modalities=["TEXT", "IMAGE"]),
        model("model-id-not-active", "model-name-3", stream_supported=True, status="DISABLED"),
        model("model-id-not-text-output", "model-name-4", stream_supported=True, output_modalities=["IMAGE"])
    )

    models = list_bedrock_models(client)

    assert models == {
        "model-id": {
            "modalities": ["TEXT"]
        }
    }

def test_one_inference_profile():
    client = FakeBedrockClient(
        inference_profile("us.model-id", "p1", "SYSTEM_DEFINED"),
        model("model-id", "model-name", stream_supported=True, input_modalities=["TEXT"])
    )

    models = list_bedrock_models(client)

    assert models == {
        "model-id": {
            "modalities": ["TEXT"]
        },
        "us.model-id": {
            "modalities": ["TEXT"]
        }
    }

def test_default_model_on_throw():
    client = ThrowingBedrockClient()

    models = list_bedrock_models(client)

    assert models == {
        "anthropic.claude-3-sonnet-20240229-v1:0": {
            "modalities": ["TEXT", "IMAGE"]
        }
    }

def inference_profile(profile_id: str, name: str, profile_type: Literal["SYSTEM_DEFINED", "APPLICATION"]):
    return {
        "inferenceProfileName": name,
        "inferenceProfileId": profile_id,
        "type": profile_type
    }

def model(
        model_id: str,
        model_name: str,
        input_modalities: list[str] = None,
        output_modalities: list[str] = None,
        stream_supported: bool = False,
        inference_types: list[str] = None,
        status: str = "ACTIVE") -> dict:
    if input_modalities is None:
        input_modalities = ["TEXT"]
    if output_modalities is None:
        output_modalities = ["TEXT"]
    if inference_types is None:
        inference_types = ["ON_DEMAND"]
    return {
                "modelArn": "arn:model:" + model_id,
                "modelId": model_id,
                "modelName": model_name,
                "providerName": "anthropic",
                "inputModalities":input_modalities,
                "outputModalities": output_modalities,
                "responseStreamingSupported": stream_supported,
                "customizationsSupported": ["FINE_TUNING"],
                "inferenceTypesSupported": inference_types,
                "modelLifecycle": {
                    "status": status
                }
            }

def _filter_inference_profiles(inference_profiles: list[dict], profile_type: Literal["SYSTEM_DEFINED", "APPLICATION"], max_results: int = 100):
    return [p for p in inference_profiles if p.get("type") == profile_type][:max_results]

def _filter_models(
        models: list[dict],
        provider_name: str | None,
        customization_type: Literal["FINE_TUNING","CONTINUED_PRE_TRAINING","DISTILLATION"] | None,
        output_modality: Literal["TEXT","IMAGE","EMBEDDING"] | None,
        inference_type: Literal["ON_DEMAND","PROVISIONED"] | None):
    return [m for m in models if
                (provider_name is None or m.get("providerName") == provider_name) and
                (output_modality is None or output_modality in m.get("outputModalities")) and
                (customization_type is None or customization_type in m.get("customizationsSupported")) and
                (inference_type is None or inference_type in m.get("inferenceTypesSupported"))
            ]

class ThrowingBedrockClient(BedrockClientInterface):
    def list_inference_profiles(self, **kwargs) -> dict:
        raise Exception("throwing bedrock client always throws exception")
    def list_foundation_models(self, **kwargs) -> dict:
        raise Exception("throwing bedrock client always throws exception")

class FakeBedrockClient(BedrockClientInterface):
    def __init__(self, *args):
        self.inference_profiles = [p for p in args if p.get("inferenceProfileId", "") != ""]
        self.models = [m for m in args if m.get("modelId", "") != ""]

        unexpected =  [u for u in args if (u.get("modelId", "") == "" and u.get("inferenceProfileId", "") == "")]
        if len(unexpected) > 0:
            raise Exception("expected a model or a profile")

    def list_inference_profiles(self, **kwargs) -> dict:
        return {
            "inferenceProfileSummaries": _filter_inference_profiles(
                                    self.inference_profiles,
                                    profile_type=kwargs["typeEquals"],
                                    max_results=kwargs.get("maxResults", 100)
                                 )
        }

    def list_foundation_models(self, **kwargs) -> dict:
        return {
            "modelSummaries": _filter_models(
                                self.models,
                                provider_name=kwargs.get("byProvider", None),
                                customization_type=kwargs.get("byCustomizationType", None),
                                output_modality=kwargs.get("byOutputModality", None),
                                inference_type=kwargs.get("byInferenceType", None)
                              )
        }