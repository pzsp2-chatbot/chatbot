class EmbeddingPipeline:
    def __init__(self, loader, embedder):
        self.loader = loader
        self.embedder = embedder

    def run(self):
        articles = self.loader.load_all()
        if not articles:
            raise RuntimeError("No articles to embed.")

        texts = [a.to_text() for a in articles]
        embeddings = self.embedder.embed(texts)
        payloads = [a.__dict__ for a in articles]
        ids = [a.id for a in articles]

        return ids, embeddings, payloads
