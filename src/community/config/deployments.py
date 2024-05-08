from enum import StrEnum

from community.model_deployments import Deployment

# from community.model_deployments.hugging_face import HuggingFaceDeployment
# from community.model_deployments.local_model import LocalModelDeployment
from community.model_deployments.ollama_platform import OllamaDeployment


class ModelDeploymentName(StrEnum):
    # HuggingFace = "HuggingFace"
    # LocalModel = "LocalModel"
    Ollama = "Ollama"


AVAILABLE_MODEL_DEPLOYMENTS = {
    # ModelDeploymentName.HuggingFace: Deployment(
    #     name=ModelDeploymentName.HuggingFace,
    #     deployment_class=HuggingFaceDeployment,
    #     models=HuggingFaceDeployment.list_models(),
    #     is_available=HuggingFaceDeployment.is_available(),
    #     env_vars=[],
    # ),
    # ModelDeploymentName.LocalModel: Deployment(
    #     name=ModelDeploymentName.LocalModel,
    #     deployment_class=LocalModelDeployment,
    #     models=LocalModelDeployment.list_models(),
    #     is_available=LocalModelDeployment.is_available(),
    #     env_vars=[],
    #     kwargs={
    #         "model_path": "path/to/model",  # Note that the model needs to be in the src directory
    #     },
    # ),
    ModelDeploymentName.Ollama: Deployment(
        name=ModelDeploymentName.Ollama,
        deployment_class=OllamaDeployment,
        models=OllamaDeployment.list_models(),
        is_available=OllamaDeployment.is_available(),
        env_vars=[],
    ),
}
