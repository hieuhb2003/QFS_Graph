import asyncio
import time
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

from ..utils.logger_config import get_logger


class BaseLLMClient(ABC):
    """Base class cho LLM clients"""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.logger = get_logger()
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text từ prompt"""
        pass
    
    @abstractmethod
    async def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text từ nhiều prompts"""
        pass


class VLLMClient(BaseLLMClient):
    """vLLM client cho self-hosted models"""
    
    def __init__(self, 
                 model_name: str,
                 url: str = "http://localhost:8000/v1",
                 api_key: str = "dummy",
                 max_retries: int = 3,
                 timeout: int = 30,
                 **kwargs):
        """
        Khởi tạo vLLM client
        
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
            self.logger.info(f"Initialized vLLM client for model: {model_name} at {url}")
        except ImportError:
            raise ImportError("OpenAI library not installed. Please install: pip install openai")
    
    async def generate(self, 
                      prompt: str, 
                      temperature: float = 0.1,
                      max_tokens: int = 2048,
                      **kwargs) -> str:
        """
        Generate text từ prompt
        
        Args:
            prompt: Input prompt
            temperature: Temperature cho generation
            max_tokens: Số tokens tối đa
            **kwargs: Additional parameters
        
        Returns:
            str: Generated text
        """
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                
                generated_text = response.choices[0].message.content
                elapsed_time = time.time() - start_time
                
                self.logger.debug(f"Generated text in {elapsed_time:.2f}s, tokens: {response.usage.total_tokens}")
                return generated_text
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    self.logger.error(f"All {self.max_retries} attempts failed for prompt")
                    raise
                await asyncio.sleep(1)  # Wait before retry
    
    async def generate_batch(self, 
                           prompts: List[str], 
                           temperature: float = 0.1,
                           max_tokens: int = 2048,
                           **kwargs) -> List[str]:
        """
        Generate text từ nhiều prompts
        
        Args:
            prompts: List of input prompts
            temperature: Temperature cho generation
            max_tokens: Số tokens tối đa
            **kwargs: Additional parameters
        
        Returns:
            List[str]: List of generated texts
        """
        tasks = []
        for prompt in prompts:
            task = self.generate(prompt, temperature, max_tokens, **kwargs)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        generated_texts = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Error generating for prompt {i}: {result}")
                generated_texts.append("")  # Empty string for failed generations
            else:
                generated_texts.append(result)
        
        return generated_texts


class OpenAIClient(BaseLLMClient):
    """OpenAI API client"""
    
    def __init__(self, 
                 model_name: str,
                 api_key: str,
                 organization: Optional[str] = None,
                 max_retries: int = 3,
                 timeout: int = 30,
                 **kwargs):
        """
        Khởi tạo OpenAI client
        
        Args:
            model_name: Tên model (e.g., gpt-4, gpt-3.5-turbo)
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
            self.logger.info(f"Initialized OpenAI client for model: {model_name}")
        except ImportError:
            raise ImportError("OpenAI library not installed. Please install: pip install openai")
    
    async def generate(self, 
                      prompt: str, 
                      temperature: float = 0.1,
                      max_tokens: int = 2048,
                      **kwargs) -> str:
        """
        Generate text từ prompt
        
        Args:
            prompt: Input prompt
            temperature: Temperature cho generation
            max_tokens: Số tokens tối đa
            **kwargs: Additional parameters
        
        Returns:
            str: Generated text
        """
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                
                generated_text = response.choices[0].message.content
                elapsed_time = time.time() - start_time
                
                self.logger.debug(f"Generated text in {elapsed_time:.2f}s, tokens: {response.usage.total_tokens}")
                return generated_text
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    self.logger.error(f"All {self.max_retries} attempts failed for prompt")
                    raise
                await asyncio.sleep(1)  # Wait before retry
    
    async def generate_batch(self, 
                           prompts: List[str], 
                           temperature: float = 0.1,
                           max_tokens: int = 2048,
                           **kwargs) -> List[str]:
        """
        Generate text từ nhiều prompts
        
        Args:
            prompts: List of input prompts
            temperature: Temperature cho generation
            max_tokens: Số tokens tối đa
            **kwargs: Additional parameters
        
        Returns:
            List[str]: List of generated texts
        """
        tasks = []
        for prompt in prompts:
            task = self.generate(prompt, temperature, max_tokens, **kwargs)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        generated_texts = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Error generating for prompt {i}: {result}")
                generated_texts.append("")  # Empty string for failed generations
            else:
                generated_texts.append(result)
        
        return generated_texts


def create_llm_client(client_type: str = "vllm", **kwargs) -> BaseLLMClient:
    """
    Factory function để tạo LLM client
    
    Args:
        client_type: Loại client ("vllm" hoặc "openai")
        **kwargs: Parameters cho client
    
    Returns:
        BaseLLMClient: LLM client instance
    """
    if client_type.lower() == "vllm":
        return VLLMClient(**kwargs)
    elif client_type.lower() == "openai":
        return OpenAIClient(**kwargs)
    else:
        raise ValueError(f"Unsupported client type: {client_type}")


# Example usage:
if __name__ == "__main__":
    async def test_llm_clients():
        # Test vLLM client
        vllm_client = create_llm_client(
            client_type="vllm",
            model_name="llama2-7b-chat",
            url="http://localhost:8000/v1",
            api_key="dummy"
        )
        
        # Test OpenAI client
        openai_client = create_llm_client(
            client_type="openai",
            model_name="gpt-3.5-turbo",
            api_key="your-api-key"
        )
        
        prompt = "Hello, how are you?"
        
        # Test vLLM
        try:
            result = await vllm_client.generate(prompt)
            print(f"vLLM result: {result}")
        except Exception as e:
            print(f"vLLM error: {e}")
        
        # Test OpenAI
        try:
            result = await openai_client.generate(prompt)
            print(f"OpenAI result: {result}")
        except Exception as e:
            print(f"OpenAI error: {e}")
    
    asyncio.run(test_llm_clients()) 