"""Azure OpenAI (Foundry MaaS) adapter.

Azure AI Foundry MaaS endpoints are OpenAI-compatible, so this is just
OpenAIProvider pointed at the Foundry base_url. Model names carry a `-dev`
suffix (the Azure marker used by routing), but the Responses-vs-Chat dispatch
keys off the prefix, so `gpt-5.1-dev` -> Responses, `gpt-4o-dev` -> Chat —
inherited from the parent with no override.

Auth: AZURE_OPENAI_API_KEY + AZURE_FOUNDRY_ENDPOINT. We append `/v1/` unless
the endpoint already ends in `/v1`; we deliberately do NOT force the classic
`/openai/deployments/...` path, which is Azure OpenAI *Service*, not Foundry
MaaS.
"""

from __future__ import annotations

import os

from .openai import OpenAIProvider


class AzureOpenAIProvider(OpenAIProvider):
    name = "azure_openai"

    def __init__(self, api_key: str | None = None, endpoint: str | None = None):
        key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        if not key:
            raise RuntimeError("AZURE_OPENAI_API_KEY is not set")
        endpoint = endpoint or os.environ.get("AZURE_FOUNDRY_ENDPOINT")
        if not endpoint:
            raise RuntimeError("AZURE_FOUNDRY_ENDPOINT is not set")

        ep = endpoint.rstrip("/")
        base_url = ep + "/" if ep.endswith("/v1") else ep + "/v1/"

        super().__init__(api_key=key, base_url=base_url, name="azure_openai")
