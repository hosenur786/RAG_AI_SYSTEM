"""
Local sentence embeddings using transformers + torch.
Replicates sentence-transformers/all-MiniLM-L6-v2 behaviour without
requiring the sentence-transformers package.

Output: L2-normalised float32 vectors of dimension 384,
compatible with cosine similarity search in VectorIndex.
"""

import torch
from transformers import AutoTokenizer, AutoModel
from vector_index import VectorIndex


class LocalEmbedding:

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", distance_metric: str = "cosine",):
        """
        Initialise the embedding model and an empty vector index.

        Args:
            model_name:      HuggingFace model ID to load locally.
            distance_metric: Distance metric for the index ('cosine' or 'euclidean').
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[LocalEmbedding] Loading {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print("[LocalEmbedding] Model ready.")

        self.store = VectorIndex(distance_metric=distance_metric, embedding_fn=self._embed_one,)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _mean_pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute attention-mask-weighted mean over token embeddings.

        Padding tokens are zeroed out via the mask so they don't
        influence the sentence-level representation.

        Args:
            last_hidden_state: shape (batch, seq_len, hidden_size)
            attention_mask:    shape (batch, seq_len)

        Returns:
            Tensor of shape (batch, hidden_size)
        """
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = (last_hidden_state * mask_expanded).sum(dim=1)
        token_counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / token_counts

    def _embed_one(self, text: str) -> list[float]:
        """
        Embed a single string and return it as a plain Python list.

        Used internally by VectorIndex to embed query strings at search time.

        Args:
            text: The string to embed.

        Returns:
            List of floats of length 384.
        """
        return self.get_embeddings([text])[0].tolist()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_embeddings(self, texts: list[str]) -> torch.Tensor:
        """
        Embed a batch of strings locally.

        Args:
            texts: One or more strings to embed.

        Returns:
            Float32 tensor of shape (N, 384), L2-normalised, on CPU.
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            output = self.model(**encoded)

        embeddings = self._mean_pool(output.last_hidden_state, encoded["attention_mask"])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu()

    def build_index(self, chunks: list[str]) -> None:
        """
        Embed all chunks in one batch and store them in the vector index.

        Bulk-embedding is more efficient than embedding one chunk at a time
        because the GPU processes the whole batch in parallel.

        Args:
            chunks: List of text strings to index.
        """
        embeddings = self.get_embeddings(chunks)
        for chunk, embedding in zip(chunks, embeddings):
            self.store.add_vector(embedding.tolist(), {"content": chunk})
        print(f"[LocalEmbedding] Indexed {len(chunks)} chunks.")

    def search(self, question: str, k: int = 3) -> list[tuple[dict, float]]:
        """
        Find the k chunks most semantically similar to the question.

        The question is embedded automatically by VectorIndex using the
        _embed_one function provided at initialisation.

        Args:
            question: Natural-language question string.
            k:        Number of results to return (default 3).

        Returns:
            List of (document_dict, distance) tuples, sorted by
            ascending cosine distance (lower = more similar).
        """
        return self.store.search(question, k=k)

    def get_context(self, question: str, k: int = 3) -> str:

            results = self.search(question, k=k)

            print("\n" + "="*80)
            print("TOP RETRIEVAL RESULTS")
            print("="*80)

            for i, (doc, score) in enumerate(results[:10]):
                print(f"\nResult {i+1}")
                print(f"Score: {score}")
                print(doc["content"][:200])
            context = "\n\n---\n\n".join(
                doc["content"]
                for doc, _ in results[:k]
            )

            sources = [
                doc["content"]
                for doc, _ in results[:k]
            ]

            return context, sources
            """
Local sentence embeddings using transformers + torch.
Replicates sentence-transformers/all-MiniLM-L6-v2 behaviour without
requiring the sentence-transformers package.

Output: L2-normalised float32 vectors of dimension 384,
compatible with cosine similarity search in VectorIndex.
"""

import torch
from transformers import AutoTokenizer, AutoModel
from vector_index import VectorIndex


class LocalEmbedding:

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", distance_metric: str = "cosine",):
        """
        Initialise the embedding model and an empty vector index.

        Args:
            model_name:      HuggingFace model ID to load locally.
            distance_metric: Distance metric for the index ('cosine' or 'euclidean').
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[LocalEmbedding] Loading {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print("[LocalEmbedding] Model ready.")

        self.store = VectorIndex(distance_metric=distance_metric, embedding_fn=self._embed_one,)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _mean_pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute attention-mask-weighted mean over token embeddings.

        Padding tokens are zeroed out via the mask so they don't
        influence the sentence-level representation.

        Args:
            last_hidden_state: shape (batch, seq_len, hidden_size)
            attention_mask:    shape (batch, seq_len)

        Returns:
            Tensor of shape (batch, hidden_size)
        """
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = (last_hidden_state * mask_expanded).sum(dim=1)
        token_counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / token_counts

    def _embed_one(self, text: str) -> list[float]:
        """
        Embed a single string and return it as a plain Python list.

        Used internally by VectorIndex to embed query strings at search time.

        Args:
            text: The string to embed.

        Returns:
            List of floats of length 384.
        """
        return self.get_embeddings([text])[0].tolist()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_embeddings(self, texts: list[str]) -> torch.Tensor:
        """
        Embed a batch of strings locally.

        Args:
            texts: One or more strings to embed.

        Returns:
            Float32 tensor of shape (N, 384), L2-normalised, on CPU.
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            output = self.model(**encoded)

        embeddings = self._mean_pool(output.last_hidden_state, encoded["attention_mask"])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu()

    def build_index(self, chunks: list[str]) -> None:
        """
        Embed all chunks in one batch and store them in the vector index.

        Bulk-embedding is more efficient than embedding one chunk at a time
        because the GPU processes the whole batch in parallel.

        Args:
            chunks: List of text strings to index.
        """
        embeddings = self.get_embeddings(chunks)
        for idx, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
             ):
                self.store.add_vector(
                     embedding.tolist(),
                         {
                                "id": idx,
                                "content": chunk
                         }
                )
        print(f"[LocalEmbedding] Indexed {len(chunks)} chunks.")

    def search(self, question: str, k: int = 3) -> list[tuple[dict, float]]:
        """
        Find the k chunks most semantically similar to the question.

        The question is embedded automatically by VectorIndex using the
        _embed_one function provided at initialisation.

        Args:
            question: Natural-language question string.
            k:        Number of results to return (default 3).

        Returns:
            List of (document_dict, distance) tuples, sorted by
            ascending cosine distance (lower = more similar).
        """
        return self.store.search(question, k=k)

    def get_context(self, question: str, k: int = 3) -> str:

            results = self.search(question, k=k)

            print("\n" + "="*80)
            print("TOP RETRIEVAL RESULTS")
            print("="*80)

            for i, (doc, score) in enumerate(results[:10]):
                print(f"\nResult {i+1}")
                print(f"Score: {score}")
                print(doc["content"][:200])

            context = "\n\n---\n\n".join(
                doc["content"] for doc, _ in results[:k]
            )

            sources = [
                doc["content"]
                for doc, _ in results[:k]
            ]

            return context, sources
