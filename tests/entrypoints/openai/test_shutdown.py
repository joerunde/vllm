import asyncio
from http import HTTPStatus
import time
from typing import AsyncIterator, Optional
from unittest import mock

import openai
import pytest
import requests
import sys

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.outputs import RequestOutput
from vllm.version import __version__ as VLLM_VERSION
from vllm.entrypoints.openai import api_server
from vllm.entrypoints.openai.api_server import VLLMServer
from vllm.utils import FlexibleArgumentParser, get_open_port

from ...utils import RemoteOpenAIServer

class StubEngine(AsyncLLMEngine):
        
        def __init__(self, *args, **kwargs) -> None:
            self._errored_with: Optional[Exception] = None

        async def generate(self, *args, **kwargs) -> AsyncIterator[RequestOutput]:
            self._errored_with = RuntimeError("oops")
            raise self._errored_with
        
        @property
        def is_running(self):
            return not self.errored
        
class StubCompletions(OpenAIServingCompletion):

    def __init__(self, *args, **kwargs) -> None:
        pass

    def create_completion(self, *args, **kwargs):
        raise RuntimeError("foo")
    
@pytest.fixture
def server_args():
    port = get_open_port()
    sys_args = ["test_command", "--port", f"{port}", "--model", "not_a_model"]
    with mock.patch.object(sys, "argv", sys_args):
        parser = FlexibleArgumentParser(
        description="test args please ignore")
        parser = make_arg_parser(parser)
        return parser.parse_args()


@pytest.mark.asyncio
async def test_shutdown_on_engine_error(server_args):
    
    stub_engine = StubEngine()
    stub_completions = StubCompletions()

    engine_args = AsyncEngineArgs.from_cli_args(server_args)

    server = VLLMServer(args=server_args, engine=stub_engine, engine_args=engine_args, openai_serving_completion=stub_completions, openai_serving_chat=None, openai_serving_embedding=None, openai_serving_tokenization=None)
    
    server_task = asyncio.create_task(server.run())

    client = openai.OpenAI(
        base_url=f"http://localhost:{server.server.config.port}/v1",
        api_key="none",
    )

    start_time = time.time()

    print("\n\n\n\n\n\n")
    print(server_task.done())

    while not server_task.done():
        try:
            response = requests.get(f"http://localhost:{server.server.config.port}/health")
            if response.status_code != 200:
                print("not started!")
                raise RuntimeError(response.status_code)
            else:
                print("\n\n\n\n\t\t WHOOOOOOOO WE STARTED \n\n\n\n\n")
                break
        except:
            print("in health handler")
            await asyncio.sleep(0.1)
            if time.time() - start_time > 5:
                assert False, "server didn't start"

    client.completions.create(model="foo", prompt="Hello, my name is")

    assert False