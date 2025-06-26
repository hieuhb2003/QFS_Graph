import asyncio
import time
from typing import List, Union, Optional
import numpy as np
from abc import ABC, abstractmethod

from ..utils.logger_config import get_logger


class BaseEmbeddingClient(ABC):
    """Base class cho embedding clients"""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.logger = get_logger()
        self.embedding_dim = None
    
    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Return embedding dimension"""
        pass
    
    @abstractmethod
    async def embed(self, texts: List[str]) -> List[np.ndarray]:
        """Embed list of texts"""
        pass
    
    @abstractmethod
    async def embed_single(self, text: str) -> np.ndarray:
        """Embed single text"""
        pass


class VLLMEmbeddingClient(BaseEmbeddingClient):
    """vLLM embedding client cho self-hosted models"""
    
    def __init__(self, 
                 model_name: str,
                 url: str = "http://localhost:8000/v1",
                 api_key: str = "dummy",
                 max_retries: int = 3,
                 timeout: int = 30,
                 **kwargs):
        """
        Khởi tạo vLLM embedding client
        
        Args:
            model_name: Tên model
            url: URL của vLLM server
            api_key: API key (có thể là dummy cho local)
            max_retries: Số lần retry tối đa
            timeout: Timeout cho request (seconds)
        """
        super().__init__(model_name)
        self.url = url
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Import OpenAI client
        try:
            import openai
            self.client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url=url
            )
            self.logger.info(f"Initialized vLLM embedding client for model: {model_name} at {url}")
        except ImportError:
            raise ImportError("OpenAI library not installed. Please install: pip install openai")
        
        # Get embedding dimension (cần test với một text)
        self._embedding_dim = None
    
    @property
    def embedding_dimension(self) -> int:
        """Return embedding dimension"""
        if self._embedding_dim is None:
            # Lazy initialization - get dimension from first embedding
            self.logger.warning("Embedding dimension not initialized. Will be set on first embedding call.")
            return 384  # Default fallback
        return self._embedding_dim
    
    async def embed(self, texts: List[str]) -> List[np.ndarray]:
        """
        Embed list of texts
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List[np.ndarray]: List of embeddings
        """
        if not texts:
            return []
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                response = await self.client.embeddings.create(
                    model=self.model_name,
                    input=texts
                )
                
                embeddings = []
                for data in response.data:
                    embedding = np.array(data.embedding, dtype=np.float32)
                    embeddings.append(embedding)
                
                # Set embedding dimension if not set
                if self._embedding_dim is None and embeddings:
                    self._embedding_dim = len(embeddings[0])
                    self.logger.info(f"Set embedding dimension to {self._embedding_dim}")
                
                elapsed_time = time.time() - start_time
                self.logger.debug(f"Embedded {len(texts)} texts in {elapsed_time:.2f}s")
                
                return embeddings
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    self.logger.error(f"All {self.max_retries} attempts failed for embedding")
                    raise
                await asyncio.sleep(1)  # Wait before retry
    
    async def embed_single(self, text: str) -> np.ndarray:
        """
        Embed single text
        
        Args:
            text: Text to embed
        
        Returns:
            np.ndarray: Embedding vector
        """
        embeddings = await self.embed([text])
        return embeddings[0] if embeddings else np.array([])


class SentenceTransformersEmbeddingClient(BaseEmbeddingClient):
    """Sentence Transformers embedding client cho GPU acceleration"""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 device: str = "cuda",
                 max_retries: int = 3,
                 **kwargs):
        """
        Khởi tạo Sentence Transformers client
        
        Args:
            model_name: Tên model (e.g., all-MiniLM-L6-v2, all-mpnet-base-v2)
            device: Device to use ("cuda", "cpu", etc.)
            max_retries: Số lần retry tối đa
        """
        super().__init__(model_name)
        self.device = device
        self.max_retries = max_retries
        
        # Import sentence transformers
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, device=device)
            self._embedding_dim = self.model.get_sentence_embedding_dimension()
            self.logger.info(f"Initialized SentenceTransformers client for model: {model_name} on {device}")
            self.logger.info(f"Embedding dimension: {self._embedding_dim}")
        except ImportError:
            raise ImportError("SentenceTransformers library not installed. Please install: pip install sentence-transformers")
    
    @property
    def embedding_dimension(self) -> int:
        """Return embedding dimension"""
        return self._embedding_dim
    
    async def embed(self, texts: List[str]) -> List[np.ndarray]:
        """
        Embed list of texts
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List[np.ndarray]: List of embeddings
        """
        if not texts:
            return []
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                # Run embedding in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                embeddings = await loop.run_in_executor(
                    None, 
                    lambda: self.model.encode(texts, convert_to_numpy=True)
                )
                
                # Convert to list of numpy arrays
                embedding_list = [np.array(emb, dtype=np.float32) for emb in embeddings]
                
                elapsed_time = time.time() - start_time
                self.logger.debug(f"Embedded {len(texts)} texts in {elapsed_time:.2f}s")
                
                return embedding_list
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    self.logger.error(f"All {self.max_retries} attempts failed for embedding")
                    raise
                await asyncio.sleep(1)  # Wait before retry
    
    async def embed_single(self, text: str) -> np.ndarray:
        """
        Embed single text
        
        Args:
            text: Text to embed
        
        Returns:
            np.ndarray: Embedding vector
        """
        embeddings = await self.embed([text])
        return embeddings[0] if embeddings else np.array([])


class OpenAIEmbeddingClient(BaseEmbeddingClient):
    """OpenAI embedding client"""
    
    def __init__(self, 
                 model_name: str = "text-embedding-ada-002",
                 api_key: str = None,
                 organization: Optional[str] = None,
                 max_retries: int = 3,
                 timeout: int = 30,
                 **kwargs):
        """
        Khởi tạo OpenAI embedding client
        
        Args:
            model_name: Tên model (e.g., text-embedding-ada-002)
            api_key: OpenAI API key
            organization: OpenAI organization ID (optional)
            max_retries: Số lần retry tối đa
            timeout: Timeout cho request (seconds)
        """
        super().__init__(model_name)
        self.api_key = api_key
        self.organization = organization
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Import OpenAI client
        try:
            import openai
            self.client = openai.AsyncOpenAI(
                api_key=api_key,
                organization=organization
            )
            self.logger.info(f"Initialized OpenAI embedding client for model: {model_name}")
        except ImportError:
            raise ImportError("OpenAI library not installed. Please install: pip install openai")
        
        # Set embedding dimension based on model
        self._embedding_dim = self._get_embedding_dimension(model_name)
    
    def _get_embedding_dimension(self, model_name: str) -> int:
        """Get embedding dimension for model"""
        dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
        return dimensions.get(model_name, 1536)  # Default to 1536
    
    @property
    def embedding_dimension(self) -> int:
        """Return embedding dimension"""
        return self._embedding_dim
    
    async def embed(self, texts: List[str]) -> List[np.ndarray]:
        """
        Embed list of texts
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List[np.ndarray]: List of embeddings
        """
        if not texts:
            return []
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                response = await self.client.embeddings.create(
                    model=self.model_name,
                    input=texts
                )
                
                embeddings = []
                for data in response.data:
                    embedding = np.array(data.embedding, dtype=np.float32)
                    embeddings.append(embedding)
                
                elapsed_time = time.time() - start_time
                self.logger.debug(f"Embedded {len(texts)} texts in {elapsed_time:.2f}s")
                
                return embeddings
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    self.logger.error(f"All {self.max_retries} attempts failed for embedding")
                    raise
                await asyncio.sleep(1)  # Wait before retry
    
    async def embed_single(self, text: str) -> np.ndarray:
        """
        Embed single text
        
        Args:
            text: Text to embed
        
        Returns:
            np.ndarray: Embedding vector
        """
        embeddings = await self.embed([text])
        return embeddings[0] if embeddings else np.array([])


def create_embedding_client(client_type: str = "sentence_transformers", **kwargs) -> BaseEmbeddingClient:
    """
    Factory function để tạo embedding client
    
    Args:
        client_type: Loại client ("vllm", "sentence_transformers", hoặc "openai")
        **kwargs: Parameters cho client
    
    Returns:
        BaseEmbeddingClient: Embedding client instance
    """
    if client_type.lower() == "vllm":
        return VLLMEmbeddingClient(**kwargs)
    elif client_type.lower() == "sentence_transformers":
        return SentenceTransformersEmbeddingClient(**kwargs)
    elif client_type.lower() == "openai":
        return OpenAIEmbeddingClient(**kwargs)
    else:
        raise ValueError(f"Unsupported client type: {client_type}")


# Example usage:
if __name__ == "__main__":
    async def test_embedding_clients():
        # Test SentenceTransformers client
        st_client = create_embedding_client(
            client_type="sentence_transformers",
            model_name="all-MiniLM-L6-v2",
            device="cuda"  # or "cpu"
        )
        
        # Test vLLM client
        vllm_client = create_embedding_client(
            client_type="vllm",
            model_name="llama2-7b-chat",
            url="http://localhost:8000/v1",
            api_key="dummy"
        )
        
        # Test OpenAI client
        openai_client = create_embedding_client(
            client_type="openai",
            model_name="text-embedding-ada-002",
            api_key="your-api-key"
        )
        
        texts = ["Hello world", "How are you?", "This is a test"]
        
        # Test SentenceTransformers
        try:
            embeddings = await st_client.embed(texts)
            print(f"SentenceTransformers embeddings shape: {[emb.shape for emb in embeddings]}")
        except Exception as e:
            print(f"SentenceTransformers error: {e}")
        
        # Test vLLM
        try:
            embeddings = await vllm_client.embed(texts)
            print(f"vLLM embeddings shape: {[emb.shape for emb in embeddings]}")
        except Exception as e:
            print(f"vLLM error: {e}")
        
        # Test OpenAI
        try:
            embeddings = await openai_client.embed(texts)
            print(f"OpenAI embeddings shape: {[emb.shape for emb in embeddings]}")
        except Exception as e:
            print(f"OpenAI error: {e}")
    
    asyncio.run(test_embedding_clients()) 