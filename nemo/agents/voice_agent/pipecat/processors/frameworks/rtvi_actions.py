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

"""Helpers that register the common RTVI client-message handlers used by voice-agent bots.

Pipecat 1.x removed ``RTVIAction`` and ``RTVIProcessor.register_action``. Custom
client-initiated commands are now plain RTVI client messages with a ``type``
string and ``data`` payload, dispatched via the ``on_client_message`` event and
replied to with ``send_server_response``.

Each ``register_*_handler`` here installs an ``on_client_message`` handler on
the supplied ``RTVIProcessor`` that filters by ``msg.type`` so multiple
helpers can coexist on the same processor.

The reset and update-prompt helpers need to queue an ``EndTaskFrame`` onto a
``PipelineTask`` that is typically created *after* the RTVI processor (because
the task needs ``rtvi`` in its observer list). ``TaskRef`` is a tiny holder the
bot sets after constructing the task.
"""

import copy
import dataclasses
import json
from typing import Any, Awaitable, Callable, List, Optional

from loguru import logger
from pipecat.frames.frames import EndTaskFrame
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frameworks.rtvi import RTVIProcessor
from pipecat.processors.frameworks.rtvi.models import ClientMessage
from pipecat.services.ai_service import AIService

CONTEXT_RESET_TYPE = "context.reset"
CONTEXT_UPDATE_SYSTEM_PROMPT_TYPE = "context.update_system_prompt"
CONTEXT_GET_HISTORY_TYPE = "context.get_context_history"


@dataclasses.dataclass
class TaskRef:
    """Mutable handle to a PipelineTask and its running flag.

    Construct early, hand to RTVI action factories, then populate once the task
    exists. ``running`` is flipped by the bot runner during shutdown so handlers
    can avoid queueing frames onto a dead task.
    """

    task: Optional[PipelineTask] = None
    running: bool = False


async def _maybe_end_task(task_ref: TaskRef) -> None:
    if task_ref.running and task_ref.task is not None:
        await task_ref.task.queue_frames([EndTaskFrame()])


def _reset_services(services: List[AIService]) -> None:
    for service in services:
        if service is not None and hasattr(service, "reset"):
            service.reset()


def _register_typed_handler(
    rtvi: RTVIProcessor,
    msg_type: str,
    handler: Callable[[ClientMessage], Awaitable[Any]],
) -> None:
    """Register an on_client_message handler that fires only for ``msg_type``.

    The handler returns the payload to send back via ``send_server_response``.
    """

    async def on_client_message(processor: RTVIProcessor, msg: ClientMessage) -> None:
        if msg.type != msg_type:
            return
        try:
            data = await handler(msg)
        except Exception as e:
            logger.error(f"Error handling RTVI message {msg_type}: {e}")
            data = {"error": str(e)}
        await processor.send_server_response(msg, data)

    rtvi.add_event_handler("on_client_message", on_client_message)


def register_reset_context_handler(
    rtvi: RTVIProcessor,
    task_ref: TaskRef,
    user_aggregator,
    assistant_aggregator,
    original_messages: List[dict],
    resettable_services: List[AIService],
) -> None:
    """Install the ``context.reset`` client-message handler on ``rtvi``.

    ``original_messages`` is captured by reference so the handler always resets
    to whatever ``register_update_system_prompt_handler`` last wrote.
    """

    async def handle(_msg: ClientMessage) -> dict:
        logger.info("Resetting conversation context...")
        await _maybe_end_task(task_ref)
        user_aggregator.reset()
        assistant_aggregator.reset()
        user_aggregator.set_messages(copy.deepcopy(original_messages))
        assistant_aggregator.set_messages(copy.deepcopy(original_messages))
        _reset_services(resettable_services)
        logger.info("Conversation context reset successfully")
        return {"result": True}

    _register_typed_handler(rtvi, CONTEXT_RESET_TYPE, handle)


def register_update_system_prompt_handler(
    rtvi: RTVIProcessor,
    task_ref: TaskRef,
    user_aggregator,
    assistant_aggregator,
    original_messages: List[dict],
    resettable_services: List[Any],
    *,
    system_role: str,
    system_prompt_suffix: str,
    enable_tool_calling: bool = False,
    llm=None,
    context=None,
    tool_factory: Optional[Callable[..., Any]] = None,
    register_schema_tools: Optional[Callable[..., Any]] = None,
) -> None:
    """Install the ``context.update_system_prompt`` client-message handler on ``rtvi``.

    Tool registration is optional. When ``enable_tool_calling`` is True and a
    ``tools`` JSON string is supplied by the caller, ``tool_factory`` is invoked
    per tool to produce schema tools, then ``register_schema_tools`` swaps them
    onto ``llm`` / ``context``.
    """

    async def handle(msg: ClientMessage) -> dict:
        await _maybe_end_task(task_ref)

        arguments = msg.data or {}
        new_prompt = arguments.get("prompt", "")
        new_tools_json = arguments.get("tools", "{}")
        if not new_prompt:
            logger.error("No prompt provided in update_system_prompt message")
            return {"result": False, "error": "missing prompt"}

        logger.info(f"Updating system prompt to: {new_prompt[:100]}...")

        if arguments.get("add_suffix", True) and system_prompt_suffix:
            new_prompt = f"{new_prompt}\n{system_prompt_suffix}"

        new_messages = [{"role": system_role, "content": new_prompt}]

        original_messages.clear()
        original_messages.extend(new_messages)

        user_aggregator.reset()
        assistant_aggregator.reset()
        user_aggregator.set_messages(copy.deepcopy(new_messages))
        assistant_aggregator.set_messages(copy.deepcopy(new_messages))

        if enable_tool_calling and new_tools_json and tool_factory is not None and register_schema_tools is not None:
            logger.info("Registering new tools...")
            new_tools = json.loads(new_tools_json)
            shared_state: dict = {}
            new_schema_tools = [
                tool_factory(tool_name, rtvi=rtvi, shared_state=shared_state, **tool_args)
                for tool_name, tool_args in new_tools.items()
            ]
            register_schema_tools(
                llm=llm,
                context=context,
                tools=new_schema_tools,
                cancel_on_interruption=False,
                keep_existing_tools=False,
            )
        else:
            logger.info(
                "Tool calling disabled, no tools provided, or tool_factory not configured; "
                "skipping tool registration."
            )

        logger.debug(f"user context tools: {user_aggregator._context.tools}")
        logger.debug(f"assistant context tools: {assistant_aggregator._context.tools}")

        _reset_services(resettable_services)

        logger.info("System prompt updated and context reset successfully")
        return {"result": True}

    _register_typed_handler(rtvi, CONTEXT_UPDATE_SYSTEM_PROMPT_TYPE, handle)


def register_get_context_history_handler(
    rtvi: RTVIProcessor,
    task_ref: TaskRef,
    assistant_aggregator,
) -> None:
    """Install the ``context.get_context_history`` client-message handler on ``rtvi``.

    Returns the assistant aggregator's full message list, stringified to match
    the shape evaluation clients expect.
    """

    async def handle(_msg: ClientMessage) -> dict:
        await _maybe_end_task(task_ref)
        try:
            messages = assistant_aggregator._context.get_messages()
            logger.debug(f"Returning context history: {len(messages)} messages")
            return {"context": str(messages)}
        except Exception as e:
            logger.error(f"Error getting context history: {e}")
            return {"context": []}

    _register_typed_handler(rtvi, CONTEXT_GET_HISTORY_TYPE, handle)


# Backward-compatible aliases for the old factory names. They no longer return
# RTVIAction objects (those don't exist in pipecat 1.x); instead they install
# the handler on the supplied processor and return ``None``. Bot scripts should
# migrate to the new names, but the aliases keep older callsites working.
def create_reset_context_action(
    task_ref: TaskRef,
    user_aggregator,
    assistant_aggregator,
    original_messages: List[dict],
    resettable_services: List[AIService],
    *,
    rtvi: Optional[RTVIProcessor] = None,
) -> None:
    if rtvi is None:
        raise TypeError("create_reset_context_action now requires rtvi=… (pipecat 1.x dropped RTVIAction)")
    register_reset_context_handler(
        rtvi, task_ref, user_aggregator, assistant_aggregator, original_messages, resettable_services
    )


def create_update_system_prompt_action(
    task_ref: TaskRef,
    user_aggregator,
    assistant_aggregator,
    original_messages: List[dict],
    resettable_services: List[Any],
    *,
    system_role: str,
    system_prompt_suffix: str,
    enable_tool_calling: bool = False,
    llm=None,
    context=None,
    rtvi: Optional[RTVIProcessor] = None,
    tool_factory: Optional[Callable[..., Any]] = None,
    register_schema_tools: Optional[Callable[..., Any]] = None,
) -> None:
    if rtvi is None:
        raise TypeError("create_update_system_prompt_action now requires rtvi=… (pipecat 1.x dropped RTVIAction)")
    register_update_system_prompt_handler(
        rtvi,
        task_ref,
        user_aggregator,
        assistant_aggregator,
        original_messages,
        resettable_services,
        system_role=system_role,
        system_prompt_suffix=system_prompt_suffix,
        enable_tool_calling=enable_tool_calling,
        llm=llm,
        context=context,
        tool_factory=tool_factory,
        register_schema_tools=register_schema_tools,
    )


def create_get_context_history_action(
    task_ref: TaskRef,
    assistant_aggregator,
    *,
    rtvi: Optional[RTVIProcessor] = None,
) -> None:
    if rtvi is None:
        raise TypeError("create_get_context_history_action now requires rtvi=… (pipecat 1.x dropped RTVIAction)")
    register_get_context_history_handler(rtvi, task_ref, assistant_aggregator)
