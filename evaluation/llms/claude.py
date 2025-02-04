"""
Claude API inference.
"""

import anthropic
from anthropic import AsyncAnthropic, Anthropic
import time
from typing import List, Dict, Optional, Union, Any, cast
import asyncio
from datetime import datetime
import json
import os
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod
from tenacity import (
    AsyncRetrying,
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)
from tqdm import trange

_MODEL_REQUIRED_MSG = "model is required"
CLAUDE_RETRY_ERROR_TYPES = (
    cast(Any, anthropic).RateLimitError,
    cast(Any, anthropic).APIConnectionError,
)


class BaseClaudeLLM(ABC):
    """The Base Claude LLM implementation."""

    _async_client: AsyncAnthropic
    _sync_client: Anthropic

    def __init__(self):
        self._create_claude_client()

    @abstractmethod
    def _create_claude_client(self):
        """Create a new synchronous and asynchronous Claude client instance."""

    def set_clients(
        self,
        sync_client: Anthropic,
        async_client: AsyncAnthropic,
    ):
        """
        Set the synchronous and asynchronous clients used for making API requests.

        Args:
            sync_client (Anthropic): The sync client object.
            async_client (AsyncAnthropic): The async client object.
        """
        self._sync_client = sync_client
        self._async_client = async_client

    @property
    def async_client(self) -> Optional[AsyncAnthropic]:
        """
        Get the asynchronous client used for making API requests.

        Returns
        -------
            AsyncOpenAI | AsyncAzureOpenAI: The async client object.
        """
        return self._async_client

    @property
    def sync_client(self) -> Optional[Anthropic]:
        """
        Get the synchronous client used for making API requests.

        Returns
        -------
            AsyncOpenAI | AsyncAzureOpenAI: The async client object.
        """
        return self._sync_client

    @async_client.setter
    def async_client(self, client: AsyncAnthropic):
        """
        Set the asynchronous client used for making API requests.

        Args:
            client (AsyncOpenAI | AsyncAzureOpenAI): The async client object.
        """
        self._async_client = client

    @sync_client.setter
    def sync_client(self, client: Anthropic):
        """
        Set the synchronous client used for making API requests.

        Args:
            client (OpenAI | AzureOpenAI): The sync client object.
        """
        self._sync_client = client



# Synchronous Version

class ChatClaude(BaseClaudeLLM):
    def __init__(
            self,
            api_key: Optional[str] = None,
            model: Optional[str] = None,
            batch_size: int = 8,
            max_retries: int = 10,
            request_timeout: float = 180.0,
            retry_error_types: tuple[type[BaseException]] = CLAUDE_RETRY_ERROR_TYPES,  # type: ignore
            **kwargs,
            ):
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.retry_error_types = retry_error_types
        self.batch_size = batch_size
        super().__init__()

    def _create_claude_client(self):
        sync_client = anthropic.Anthropic(api_key=self.api_key)
        async_client = anthropic.AsyncAnthropic(api_key=self.api_key)
        self.set_clients(sync_client=sync_client, async_client=async_client)



    def create_single_input(self, system_prompt, user_content: str):

        message =(system_prompt,
                  [
            {"role": "user", "content": user_content},
                    ]
                  )

        return message

    def _generate(
        self,
        messages: Union[str, list[Any]],
        **kwargs: Any,
    ) -> str:
        model = self.model
        if not model:
            raise ValueError(_MODEL_REQUIRED_MSG)
        response = self.sync_client.messages.create(
                model=model,
                max_tokens=2048,
                system=messages[0],
                messages=messages[1],
            )
        return response.content[0].text or ""  # type: ignore


    async def _agenerate(
        self,
        messages: Union[str, list[Any]],
        **kwargs: Any,
    ) -> str:
        model = self.model
        if not model:
            raise ValueError(_MODEL_REQUIRED_MSG)
        response = await self.async_client.messages.create(  # type: ignore
            model=model,
            max_tokens=2048,
            system=messages[0],
            messages=messages[1],  # type: ignore
            **kwargs,
        )
        return response.content[0].text or ""  # type: ignore

    def generate(
        self,
        messages: Union[str, list[Any]],
        **kwargs: Any,
    ) -> dict:
        """Generate text."""

        try:
            retryer = Retrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential_jitter(max=10),
                reraise=True,
                retry=retry_if_exception_type(self.retry_error_types),
            )
            for attempt in retryer:
                with attempt:
                    response = self._generate(
                        messages=messages,
                        **kwargs,
                    )
                    return {
                        "query": messages[1][0]["content"],
                        "response": response,
                        "timestamp": datetime.now().isoformat(),
                        "status": "success"
                    }

        except RetryError as e:
            return {
                "query": messages[1][0]["content"],
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "error"
            }
        else:
            # TODO: why not just throw in this case?
            return ""


    async def agenerate(
        self,
        messages: Union[str, list[Any]],
        **kwargs: Any,
    ) -> dict:
        """Generate text asynchronously."""
        try:
            retryer = AsyncRetrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential_jitter(max=10),
                reraise=True,
                retry=retry_if_exception_type(self.retry_error_types),  # type: ignore
            )
            async for attempt in retryer:
                with attempt:
                    response = await self._agenerate(
                        messages=messages,
                        **kwargs,
                    )
                    return {
                        "query": messages[1][0]["content"],
                        "response": response,
                        "timestamp": datetime.now().isoformat(),
                        "status": "success"
                    }
        except RetryError as e:
            return {
                "query": messages[1][0]["content"],
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "error"
            }
        else:
            # TODO: why not just throw in this case?
            return ""

    def batch_inference(self, system_prompt, batch_input):
        batch_input = [self.create_single_input(system_prompt, input_text) for input_text in batch_input]
        responses = [self.generate(message) for message in batch_input]
        return responses

    async def abatch_inference(self, system_prompt, batch_input):
        batch_input = [self.create_single_input(system_prompt, input_text) for input_text in batch_input]
        responses = await asyncio.gather(*[self.agenerate(message) for message in batch_input])
        return responses


    def inference(self, system_prompt, user_contents):
        results = []
        for start_index in trange(0, len(user_contents), self.batch_size, desc=f"Batches", disable=False, ):
            batch_contents = user_contents[start_index: start_index + self.batch_size]
            batch_result = asyncio.run(self.abatch_inference(system_prompt, batch_contents))
            results.extend(batch_result)
        return results

    def save_results(self, results: List[Dict], filename: str):
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

#
#
# class ClaudeBatchProcessor:
#     def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
#         self.client = anthropic.Client(api_key=api_key)
#         self.model = model
#         self.rate_limit_delay = 1
#
#     def process_single_query(self, query: str, system_prompt: str = None) -> Dict:
#         try:
#             message = self.client.messages.create(
#                 model=self.model,
#                 max_tokens=1024,
#                 system=system_prompt,
#                 messages=[
#                     {"role": "user", "content": query}
#                 ]
#             )
#
#             return {
#                 "query": query,
#                 "response": message.content[0].text,
#                 "timestamp": datetime.now().isoformat(),
#                 "status": "success"
#             }
#
#         except Exception as e:
#             return {
#                 "query": query,
#                 "error": str(e),
#                 "timestamp": datetime.now().isoformat(),
#                 "status": "error"
#             }
#
#     def process_batch(self, queries: List[str], system_prompt: str = None) -> List[Dict]:
#         results = []
#         for query in queries:
#             result = self.process_single_query(query, system_prompt)
#             results.append(result)
#             time.sleep(self.rate_limit_delay)
#         return results
#
#     def save_results(self, results: List[Dict], filename: str):
#         with open(filename, 'w') as f:
#             json.dump(results, f, indent=2)
#
#
# class AsyncClaudeBatchProcessor:
#     def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022", max_workers: int = 3):
#         self.client = anthropic.Client(api_key=api_key)
#         self.model = model
#         self.rate_limit_delay = 1
#         self.max_workers = max_workers
#
#     def process_single_query(self, query: str, system_prompt: str = None) -> Dict:
#         try:
#             message = self.client.messages.create(
#                 model=self.model,
#                 max_tokens=1024,
#                 system=system_prompt,
#                 messages=[
#                     {"role": "user", "content": query}
#                 ]
#             )
#             time.sleep(self.rate_limit_delay)  # Rate limiting
#
#             return {
#                 "query": query,
#                 "response": message.content[0].text,
#                 "timestamp": datetime.now().isoformat(),
#                 "status": "success"
#             }
#
#         except Exception as e:
#             return {
#                 "query": query,
#                 "error": str(e),
#                 "timestamp": datetime.now().isoformat(),
#                 "status": "error"
#             }
#
#     def process_batch(self, queries: List[str], system_prompt: str = None) -> List[Dict]:
#         with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
#             futures = [
#                 executor.submit(self.process_single_query, query, system_prompt)
#                 for query in queries
#             ]
#             results = [future.result() for future in futures]
#         return results
#
#     def save_results(self, results: List[Dict], filename: str):
#         with open(filename, 'w') as f:
#             json.dump(results, f, indent=2)
