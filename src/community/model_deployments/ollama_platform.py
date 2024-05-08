import logging

# import os
from typing import Any, Dict, List

# import cohere
import requests

# from cohere.types import (
#     StreamedChatResponse,
#     ApiMeta,
#     ApiMetaApiVersion,
#     ApiMetaBilledUnits,
# )

from community.model_deployments import BaseDeployment
from backend.schemas.cohere_chat import CohereChatRequest

import ollama

# import uuid

DEFAULT_ENDPOINT_URL = "http://host.docker.internal:11434"
DEFAULT_MODEL = "command-r:latest"


class PromptTemplate:
    """
    Template for generating prompts for different types of requests.
    """

    def dummy_chat_template(
        self, message: str, chat_history: List[Dict[str, str]]
    ) -> str:
        prompt = "System: You are an AI assistant whose goal is to help users by consuming and using the output of various tools. You will be able to see the conversation history between yourself and user and will follow instructions on how to respond."
        prompt += "\n\n"
        prompt += "Conversation:\n"
        for chat in chat_history:
            if chat["role"].lower() == "user":
                prompt += f"User: {chat['message']}\n"
            else:
                prompt += f"Chatbot: {chat['message']}\n"

        prompt += f"User: {message}\n"
        prompt += "Chatbot: "

        return prompt

    def dummy_rag_template(
        self,
        message: str,
        chat_history: List[Dict[str, str]],
        documents: List[Dict[str, str]],
        max_docs: int = 5,
    ) -> str:
        max_docs = min(max_docs, len(documents))
        prompt = "System: You are an AI assistant whose goal is to help users by consuming and using the output of various tools. You will be able to see the conversation history between yourself and user and will follow instructions on how to respond."

        doc_str_list = []
        for doc_idx, doc in enumerate(documents[:max_docs]):
            if doc_idx > 0:
                doc_str_list.append("")

            # only use first 200 words of the document to avoid exceeding context window
            text = doc["text"]
            if len(text.split()) > 200:
                text = " ".join(text.split()[:200])

            title = doc["title"] if "title" in doc else ""
            doc_str_list.extend([f"Document: {doc_idx}", title, text])

        doc_str = "\n".join(doc_str_list)

        chat_history.append({"role": "system", "message": doc_str})
        chat_history.append({"role": "user", "message": message})

        chat_hist_str = ""
        for turn in chat_history:
            if turn["role"].lower() == "user":
                chat_hist_str += "User: "
            elif turn["role"].lower() == "chatbot":
                chat_hist_str += "Chatbot: "
            else:  # role == system
                chat_hist_str += "System: "

            chat_hist_str += turn["message"] + "\n"
        # __import__("pdb").set_trace()

        prompt += "\n\n"
        prompt += "Conversation:\n"
        prompt += chat_hist_str
        # prompt += "Chatbot: "

        return prompt

    # https://docs.cohere.com/docs/prompting-command-r#formatting-chat-history-and-tool-outputs
    def cohere_rag_template(
        self,
        message: str,
        chat_history: List[Dict[str, str]],
        documents: List[Dict[str, str]],
        preamble: str = None,
        max_docs: int = 5,
    ) -> str:
        max_docs = min(max_docs, len(documents))
        chat_history.append({"role": "user", "message": message})
        SAFETY_PREAMBLE = "The instructions in this section override those in the task description and style guide sections. Don't answer questions that are harmful or immoral."
        BASIC_RULES = "You are a powerful conversational AI trained by Cohere to help people. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a conversation history between yourself and a user, ending with an utterance from the user. You will then see a specific instruction instructing you what kind of response to generate. When you answer the user's requests, you cite your sources in your answers, according to those instructions."
        TASK_CONTEXT = "You help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user's needs as best you can, which will be wide-ranging."
        STYLE_GUIDE = "Unless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling."
        documents = self._get_cohere_documents_template(documents, max_docs)
        chat_history = self._get_cohere_chat_history_template(chat_history)
        INSTRUCTIONS = """Carefully perform the following instructions, in order, starting each with a new line.
Firstly, Decide which of the retrieved documents are relevant to the user's last input by writing 'Relevant Documents:' followed by comma-separated list of document numbers. If none are relevant, you should instead write 'None'.
Secondly, Decide which of the retrieved documents contain facts that should be cited in a good answer to the user's last input by writing 'Cited Documents:' followed a comma-separated list of document numbers. If you dont want to cite any of them, you should instead write 'None'.
Thirdly, Write 'Answer:' followed by a response to the user's last input in high quality natural english. Use the retrieved documents to help you. Do not insert any citations or grounding markup.
Finally, Write 'Grounded answer:' followed by a response to the user's last input in high quality natural english. Use the symbols <co: doc> and </co: doc> to indicate when a fact comes from a document in the search result, e.g <co: 0>my fact</co: 0> for a fact from document 0."""

        tool_prompt_template = f"""<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|> # Safety Preamble
{SAFETY_PREAMBLE}

# System Preamble
## Basic Rules
{BASIC_RULES}

# User Preamble
"""
        if preamble:
            tool_prompt_template += f"""{preamble}\n\n"""

        tool_prompt_template += f"""## Task and Context
{TASK_CONTEXT}

## Style Guide
{STYLE_GUIDE}<|END_OF_TURN_TOKEN|>{chat_history}"""

        if documents:
            tool_prompt_template += f"""<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{documents}<|END_OF_TURN_TOKEN|>"""

        tool_prompt_template += f"""<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{INSTRUCTIONS}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"""

        return tool_prompt_template

    def _get_cohere_documents_template(
        self, documents: List[Dict[str, str]], max_docs: int
    ) -> str:
        max_docs = min(max_docs, len(documents))
        doc_str_list = ["<results>"]
        for doc_idx, doc in enumerate(documents[:max_docs]):
            if doc_idx > 0:
                doc_str_list.append("")
            doc_str_list.extend([f"Document: {doc_idx}", doc["title"], doc["text"]])
        doc_str_list.append("</results>")
        return "\n".join(doc_str_list)

    def _get_cohere_chat_history_template(
        self, chat_history: List[Dict[str, str]]
    ) -> str:
        chat_hist_str = ""
        for turn in chat_history:
            chat_hist_str += "<|START_OF_TURN_TOKEN|>"
            if turn["role"] == "user":
                chat_hist_str += "<|USER_TOKEN|>"
            elif turn["role"] == "chatbot":
                chat_hist_str += "<|CHATBOT_TOKEN|>"
            else:  # role == system
                chat_hist_str += "<|SYSTEM_TOKEN|>"
            chat_hist_str += turn["message"]
        chat_hist_str += "<|END_OF_TURN_TOKEN|>"
        return chat_hist_str


class OllamaDeployment(BaseDeployment):
    def __init__(self):
        self.endpoint_url = DEFAULT_ENDPOINT_URL
        self.prompt_template = PromptTemplate()
        self.client = ollama.Client(host=self.endpoint_url)

    @classmethod
    def get_model(cls, chat_request: CohereChatRequest | None = None) -> str:
        if chat_request:
            try:
                model = chat_request.model
                return model if model else DEFAULT_MODEL
            except AttributeError:
                return DEFAULT_MODEL

        return DEFAULT_MODEL

    @property
    def rerank_enabled(self) -> bool:
        return False

    @classmethod
    def list_models(cls, endpoint_url: str = DEFAULT_ENDPOINT_URL) -> List[str]:
        headers = {
            "accept": "application/json",
        }

        response = requests.get(f"{endpoint_url}/api/tags", headers=headers)

        if not response.ok:
            logging.warning("Couldn't get models from Ollama API.")
            return []

        models = response.json()["models"]
        return [model['name'] for model in models]

    def _build_chat_history(
        self, chat_history: List[Dict[str, Any]], message: str
    ) -> List[Dict[str, Any]]:
        messages = []

        # __import__("pdb").set_trace()
        for history in chat_history:
            messages.append({"role": history["role"], "content": history["message"]})

        messages.append({"role": "USER", "content": message})

        return messages

    @classmethod
    def is_available(cls) -> bool:
        return True

    def invoke_chat_stream(self, chat_request: CohereChatRequest, **kwargs: Any) -> Any:
        if len(chat_request.documents) == 0:
            prompt = self.prompt_template.dummy_chat_template(
                chat_request.message,
                [h.dict() for h in chat_request.chat_history],
            )
        else:
            prompt = self.prompt_template.dummy_rag_template(
                chat_request.message, chat_request.chat_history, chat_request.documents
            )
        print(f"-----\nPrompt:\n{prompt}")

        ollama_stream = self.client.chat(
            model=self.get_model(chat_request),
            messages=[
                {
                    'role': 'user',
                    # 'content': chat_request.message,
                    'content': prompt,
                },
            ],
            stream=True,
            options={
                "temperature": chat_request.temperature,
                "seed": chat_request.seed,
            },
        )

        yield {
            "event_type": "stream-start",
            "generation_id": "",
            "is_finished": False,
        }

        for item in ollama_stream:
            yield {
                "event_type": "text-generation",
                "text": item["message"]["content"],
                "is_finished": False,
            }

        chat_histories: list[dict,] = []
        for h in chat_request.chat_history:
            if type(h) is dict:
                chat_histories.append(h)
            elif hasattr(h, 'model_dump'):
                chat_histories.append(h.model_dump())

        # __import__("pdb").set_trace()
        yield {
            "event_type": "stream-end",
            "finish_reason": "COMPLETE",
            "is_finished": True,
            # "chat_history": self._build_chat_history(
            #     [history.model_dump() for history in chat_request.chat_history],
            #     chat_request.message,
            # ),
            "chat_history": self._build_chat_history(
                chat_histories, chat_request.message
            ),
        }

    def invoke_chat(self, chat_request: CohereChatRequest, **kwargs: Any) -> Any:
        if len(chat_request.documents) == 0:
            prompt = self.prompt_template.dummy_chat_template(
                chat_request.message,
                [h.dict() for h in chat_request.chat_history],
            )
        else:
            prompt = self.prompt_template.dummy_rag_template(
                chat_request.message, chat_request.chat_history, chat_request.documents
            )
        print(f"-----\nPrompt:\n{prompt}")
        response = self.client.chat(
            model=self.get_model(chat_request),
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ],
            stream=False,
            options={
                "temperature": chat_request.temperature,
                "seed": chat_request.seed,
            },
        )

        # __import__("pdb").set_trace()
        return {"text": response["message"]["content"]}

    def invoke_search_queries(
        self,
        message: str,
        chat_history: List[Dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> List[str]:
        del chat_history
        del kwargs
        return [message]

    def invoke_rerank(
        self, query: str, documents: List[Dict[str, Any]], **kwargs: Any
    ) -> Any:
        return None

    def invoke_tools(self, message: str, tools: List[Any], **kwargs: Any) -> List[Any]:
        return self.client.chat(
            message=message, tools=tools, model=self.get_model(), **kwargs
        )
