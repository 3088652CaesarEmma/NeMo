# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Optional

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frameworks.rtvi import RTVIProcessor, RTVIServerMessage, RTVITextMessageData
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService

from nemo.agents.voice_agent.evaluation.tools import register_schema_tool_for_eval
from nemo.agents.voice_agent.utils.tool_calling import StandardSchemaTool

FINAL_RESPONSE_START_TAG = "<final_response>"
FINAL_RESPONSE_END_TAG = "</final_response>"


@register_schema_tool_for_eval
class SendRTVIMessageTool(StandardSchemaTool):
    """
    Send a scenario finished message to the evaluator.
    """

    DESCRIPTION: str = """
        Send a message to the RTVI client.
        """

    def __init__(self, *, description: Optional[str] = None, rtvi: Optional[RTVIProcessor] = None):
        if description is None:
            description = self.DESCRIPTION
        if rtvi is None:
            raise ValueError("RTVI processor is required to initialize the tool")
        super().__init__(description=description)
        self._rtvi = rtvi

    @property
    def properties(self) -> Dict[str, Any]:
        """
        Return the properties for the tool.
        """
        return {
            "message": {
                "type": "string",
                "description": "The message to be sent in the required format.",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        """
        Return the required properties for the tool.
        """
        return ["message"]

    async def _execute(self, params: FunctionCallParams) -> None:
        """
        Send a message to the RTVI client.
        """
        message = params.arguments.get("message")
        message = RTVIServerMessage(data=RTVITextMessageData(text=message))
        await self._rtvi.push_transport_message(message, exclude_none=True)
        await params.result_callback({"success": True, "message": "message sent to the RTVIclient."})


@register_schema_tool_for_eval
class SendScenarioSummaryTool(SendRTVIMessageTool):
    """
    Send a "Scnario Summary" message to the RTVI client after the user has no more requests
    and the agent has answered all the user's questions The input message should contain all required information
    in the required format.
    """

    def __init__(self, rtvi: RTVIProcessor):
        description = """
        Send a "Scnario Summary" message to the RTVI client after the user has no more requests 
        and the agent has answered all the user's questions The input message should contain all required information 
        in the required format.
        """
        super().__init__(description=description, rtvi=rtvi)

    async def _execute(self, params: FunctionCallParams) -> None:
        """
        Send a "Scnario Summary" message to the client, which
        should contain all required information for the evaluation.
        """
        message = params.arguments.get("message")
        message = f"{FINAL_RESPONSE_START_TAG}{message}{FINAL_RESPONSE_END_TAG}"
        logger.debug(f"Sending final response message: {message}")
        params.arguments["message"] = message
        await super()._execute(params)
