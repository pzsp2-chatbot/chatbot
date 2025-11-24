from embeddings.interfaces.embedder import IEmbedder
from openai import OpenAI

class OpenAIEmbedder(IEmbedder):
    def __init__(self, client, model="text-embedding-3-small"):
        self.client = client
        self.model = model

    def embed(self, texts):
        try:
            response = self.client.embeddings.create(model=self.model, input=texts)
            return [x.embedding for x in response.data]
        except Exception as e:
            raise RuntimeError(f"OpenAI embedding failed: {e}")

