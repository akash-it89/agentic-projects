import boto3


class BedrockLLM:
    """Amazon Bedrock Converse API. Credentials come from the normal AWS chain (`aws configure`)."""

    def __init__(self, model_id: str, region: str):
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id
        self.name = f"bedrock:{model_id}"

    def complete(self, system: str, user: str, max_tokens: int = 300) -> str:
        resp = self.client.converse(
            modelId=self.model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"temperature": 0, "maxTokens": max_tokens},
        )
        parts = resp["output"]["message"]["content"]
        return "".join(p.get("text", "") for p in parts).strip()
