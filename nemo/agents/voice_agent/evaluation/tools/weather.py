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


@register_schema_tool_for_eval
class GetCityWeatherTool(StandardSchemaTool):
    """
    Get the weather of a city.
    """

    DESCRIPTION: str = """
        Get the weather of a city. You need to provide the city name to get the weather.
        """

    def __init__(self, *, description: Optional[str] = None):
        if description is None:
            description = self.DESCRIPTION
        super().__init__(description=description)

    @property
    def properties(self) -> Dict[str, Any]:
        """
        Return the properties for the tool.
        """
        return {
            "city_name": {
                "type": "string",
                "description": "The name of the city to get the weather of.",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        """
        Return the required properties for the tool.
        """
        return ["city_name"]

    async def _execute(self, params: FunctionCallParams) -> None:
        """
        Get the weather of a city.
        """
        city_name = params.arguments.get("city_name")
        results = {
            "city": city_name,
            "weather": "sunny",
            "temperature": "20 degrees Celsius",
            "uv_index": "low",
        }
        await params.result_callback(results)
