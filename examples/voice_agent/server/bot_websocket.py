# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


import asyncio
import copy
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from loguru import logger
from omegaconf import OmegaConf
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndTaskFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frameworks.rtvi import RTVIAction, RTVIConfig, RTVIObserverParams, RTVIProcessor
from pipecat.serializers.protobuf import ProtobufFrameSerializer

from nemo.agents.voice_agent.pipecat.processors.frameworks.rtvi import RTVIObserver
from nemo.agents.voice_agent.pipecat.services.nemo.audio_logger import AudioLogger, RTVIAudioLoggerObserver
from nemo.agents.voice_agent.pipecat.services.nemo.diar import NemoDiarService
from nemo.agents.voice_agent.pipecat.services.nemo.llm import get_llm_service_from_config
from nemo.agents.voice_agent.pipecat.services.nemo.stt import ASR_EOU_MODELS, NemoSTTService
from nemo.agents.voice_agent.pipecat.services.nemo.tts import get_tts_service_from_config
from nemo.agents.voice_agent.pipecat.services.nemo.turn_taking import NeMoTurnTakingService
from nemo.agents.voice_agent.pipecat.transports.network.websocket_server import (
    WebsocketServerParams,
    WebsocketServerTransport,
)
from nemo.agents.voice_agent.utils.config_manager import ConfigManager
from nemo.agents.voice_agent.utils.tool_calling.basic_tools import tool_get_city_weather
from nemo.agents.voice_agent.utils.tool_calling.mixins import register_direct_tools_to_llm


async def run_bot_websocket_server(
    server_config_path: str = None,
    host: str = None,
    port: int = None,
    log_level: str = "DEBUG",
    log_file: str = "bot_server.log",
    use_fastapi: bool = False,
):
    """
    Creates a websocket server that runs indefinitely until manually stopped (Ctrl+C)
    Args:
        server_config_path: Path to the server configuration file, defaults to the value of the `SERVER_CONFIG_PATH` environment variable
        host: Host to bind the server to, defaults to the value of the `SERVER_HOST` environment variable
        port: Port to bind the server to, defaults to the value of the `WEBSOCKET_PORT` environment variable
        log_level: Log level to use, defaults to `DEBUG`
        log_file: Log file to write to, defaults to `bot_server.log`
    """
    # Load environment variables
    load_dotenv(override=True)
    if server_config_path is None:
        server_config_path = os.environ.get("SERVER_CONFIG_PATH", None)
    if host is None:
        host = os.getenv("SERVER_HOST", "0.0.0.0")
    if port is None:
        if use_fastapi:
            port = int(os.getenv("FASTAPI_PORT", 8766))
            logger.info(f"Starting FastAPI server on {host}:{port} with server config path: {server_config_path}")
            raise NotImplementedError(
                "FastAPI server is not supported yet"
            )  # TODO: [heh] add FastAPI transport support
        else:
            port = int(os.getenv("WEBSOCKET_PORT", 8765))
            logger.info(f"Starting websocket server on {host}:{port} with server config path: {server_config_path}")

    logger.info(f"Server configured to run indefinitely with no timeouts, use Ctrl+C to quit.")

    config_manager = ConfigManager(server_base_path=os.path.dirname(__file__), server_config_path=server_config_path)
    server_config = config_manager.get_server_config()
    logger.info(f"Server config: {OmegaConf.to_container(server_config, resolve=True)}")

    def setup_logging():
        # Configure loguru to output to both console and file
        logger.remove()  # Remove default handler
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=log_level,
        )

        logger.add(log_file, rotation="1 day", level=log_level)

    setup_logging()

    # Access configuration parameters from ConfigManager
    SAMPLE_RATE = config_manager.SAMPLE_RATE
    SYSTEM_PROMPT = config_manager.SYSTEM_PROMPT
    SYSTEM_ROLE = config_manager.SYSTEM_ROLE

    # Transport configuration
    TRANSPORT_AUDIO_OUT_10MS_CHUNKS = config_manager.TRANSPORT_AUDIO_OUT_10MS_CHUNKS
    RECORD_AUDIO_DATA = server_config.transport.get("record_audio_data", False)
    AUDIO_LOG_DIR = server_config.transport.get("audio_log_dir", "./audio_logs")

    # VAD configuration
    vad_params = config_manager.get_vad_params()

    # STT configuration
    STT_MODEL = config_manager.STT_MODEL
    STT_DEVICE = config_manager.STT_DEVICE
    stt_params = config_manager.get_stt_params()

    # Diarization configuration
    DIAR_MODEL = config_manager.DIAR_MODEL
    USE_DIAR = config_manager.USE_DIAR
    diar_params = config_manager.get_diar_params()

    # Turn taking configuration
    TURN_TAKING_BACKCHANNEL_PHRASES_PATH = config_manager.TURN_TAKING_BACKCHANNEL_PHRASES_PATH
    TURN_TAKING_MAX_BUFFER_SIZE = config_manager.TURN_TAKING_MAX_BUFFER_SIZE
    TURN_TAKING_BOT_STOP_DELAY = config_manager.TURN_TAKING_BOT_STOP_DELAY

    # TTS configuration
    TTS_TYPE = config_manager.server_config.tts.type

    logger.info("Initializing WebSocket server transport...")
    logger.info("Server configured to run indefinitely with no timeouts")
    # Initialize AudioLogger if recording is enabled
    audio_logger = None
    if RECORD_AUDIO_DATA:
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        audio_logger = AudioLogger(
            log_dir=AUDIO_LOG_DIR,
            session_id=session_id,
            enabled=True,
        )
        logger.info(f"AudioLogger initialized for session: {session_id} at {AUDIO_LOG_DIR}")

    vad_analyzer = SileroVADAnalyzer(
        sample_rate=SAMPLE_RATE,
        params=vad_params,
    )
    logger.info("VAD analyzer initialized")

    has_turn_taking = True if STT_MODEL in ASR_EOU_MODELS else False
    logger.info(f"Setting STT service has_turn_taking to `{has_turn_taking}` based on model name: `{STT_MODEL}`")

    ws_transport = WebsocketServerTransport(
        params=WebsocketServerParams(
            serializer=ProtobufFrameSerializer(),
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=vad_analyzer,
            session_timeout=None,  # Disable session timeout
            audio_in_sample_rate=SAMPLE_RATE,
            can_create_user_frames=TURN_TAKING_BACKCHANNEL_PHRASES_PATH is None
            or not has_turn_taking,  # if backchannel phrases are disabled, we can use VAD to interrupt the bot immediately
            audio_out_10ms_chunks=TRANSPORT_AUDIO_OUT_10MS_CHUNKS,
        ),
        host=host,
        port=port,
    )

    logger.info("Initializing STT service...")

    stt = NemoSTTService(
        model=STT_MODEL,
        device=STT_DEVICE,
        params=stt_params,
        sample_rate=SAMPLE_RATE,
        audio_passthrough=True,
        has_turn_taking=has_turn_taking,
        backend="legacy",
        decoder_type="rnnt",
        audio_logger=audio_logger,
    )
    logger.info("STT service initialized")

    if USE_DIAR:
        diar = NemoDiarService(
            model=DIAR_MODEL,
            device=STT_DEVICE,
            params=diar_params,
            sample_rate=SAMPLE_RATE,
            backend="legacy",
            enabled=USE_DIAR,
        )
        logger.info("Diarization service initialized")
    else:
        diar = None

    turn_taking = NeMoTurnTakingService(
        use_vad=True,
        use_diar=USE_DIAR,
        max_buffer_size=TURN_TAKING_MAX_BUFFER_SIZE,
        bot_stop_delay=TURN_TAKING_BOT_STOP_DELAY,
        backchannel_phrases=TURN_TAKING_BACKCHANNEL_PHRASES_PATH,
        audio_logger=audio_logger,
    )
    logger.info("Turn taking service initialized")

    if TTS_TYPE == "nemo":
        tts = get_tts_service_from_config(config_manager.server_config.tts, audio_logger)
    else:
        raise ValueError(f"Invalid TTS type: {TTS_TYPE}")

    logger.info("TTS service initialized")

    # Setup logging again to avoid logger from being overwritten during setting up the pipeline components
    setup_logging()

    # Put LLM in the end of model initialization to reduce the chance of running out of HBM memory
    logger.info("Initializing LLM service...")
    llm = get_llm_service_from_config(server_config.llm)
    logger.info("LLM service initialized")

    messages = [
        {
            "role": SYSTEM_ROLE,
            "content": SYSTEM_PROMPT,
        }
    ]
    inject_dummy_user_message = server_config.llm.get("inject_dummy_user_message", False)
    if inject_dummy_user_message:
        messages.append(
            {
                "role": "user",
                "content": "Hello, who are you?",
            }
        )
    context = OpenAILLMContext(messages=messages)

    if server_config.llm.get("enable_tool_calling", False):
        logger.info("Tools calling for LLM is enabled by config, registering tools...")
        register_direct_tools_to_llm(llm=llm, context=context, tool_mixins=[tts], tools=[tool_get_city_weather])
    else:
        logger.info("Tools calling for LLM is disabled by config, skipping tool registration.")

    original_messages = copy.deepcopy(context.get_messages())
    original_context = copy.deepcopy(context)
    original_context.set_llm_adapter(llm.get_llm_adapter())

    context_aggregator = llm.create_context_aggregator(context)
    user_context_aggregator = context_aggregator.user()
    assistant_context_aggregator = context_aggregator.assistant()

    # RTVI events for Pipecat client UI
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    # Add reset action to RTVI processor
    async def reset_context_handler(rtvi_processor: RTVIProcessor, service: str, arguments: dict[str, any]) -> bool:
        """Reset both user and assistant context aggregators"""
        logger.info("Resetting conversation context...")
        try:
            user_context_aggregator.reset()
            assistant_context_aggregator.reset()
            user_context_aggregator.set_messages(copy.deepcopy(original_messages))
            assistant_context_aggregator.set_messages(copy.deepcopy(original_messages))
            tts.reset()
            if diar is not None:
                diar.reset()
            logger.info("Conversation context reset successfully")
            return True
        except Exception as e:
            logger.error(f"Error resetting context: {e}")
            return False

    reset_action = RTVIAction(
        service="context",
        action="reset",
        result="bool",
        arguments=[],
        handler=reset_context_handler,
    )
    rtvi.register_action(reset_action)

    # Add update_system_prompt action for dynamic prompt updates
    async def update_system_prompt_handler(
        rtvi_processor: RTVIProcessor, service: str, arguments: dict[str, any]
    ) -> bool:
        """Update the system prompt dynamically and reset the conversation"""
        try:
            new_prompt = arguments.get("prompt", "")
            if not new_prompt:
                logger.error("No prompt provided in update_system_prompt action")
                return False

            logger.info(f"Updating system prompt to: {new_prompt[:100]}...")

            # Create new messages with updated system prompt
            new_messages = [
                {
                    "role": SYSTEM_ROLE,
                    "content": new_prompt,
                }
            ]

            # Store the new messages in original_messages so reset will use them
            original_messages.clear()
            original_messages.extend(new_messages)

            # Reset the context (this will apply the new prompt)
            user_context_aggregator.reset()
            assistant_context_aggregator.reset()
            user_context_aggregator.set_messages(copy.deepcopy(new_messages))
            assistant_context_aggregator.set_messages(copy.deepcopy(new_messages))

            # Reset TTS and diarization states
            tts.reset()
            if diar is not None:
                diar.reset()

            logger.info("System prompt updated and context reset successfully")
            return True
        except Exception as e:
            logger.error(f"Error updating system prompt: {e}")
            return False

    update_prompt_action = RTVIAction(
        service="context",
        action="update_system_prompt",
        result="bool",
        arguments=[
            {
                "name": "prompt",
                "type": "string",
                "required": True,
            }
        ],
        handler=update_system_prompt_handler,
    )
    rtvi.register_action(update_prompt_action)

    logger.info("Setting up pipeline...")

    pipeline = [
        ws_transport.input(),
        rtvi,
        stt,
    ]

    if USE_DIAR:
        pipeline.append(diar)

    pipeline.extend(
        [turn_taking, user_context_aggregator, llm, tts, ws_transport.output(), assistant_context_aggregator]
    )

    pipeline = Pipeline(pipeline)

    rtvi_params = RTVIObserverParams(bot_llm_enabled=False)
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=False,
            enable_usage_metrics=False,
            send_initial_empty_metrics=True,
            report_only_initial_ttfb=True,
            idle_timeout=None,  # Disable idle timeout
        ),
        observers=[
            RTVIObserver(rtvi, params=rtvi_params),
            RTVIAudioLoggerObserver(audio_logger=audio_logger),
        ],
        idle_timeout_secs=None,
        cancel_on_idle_timeout=False,
    )

    # Track task state
    task_running = True

    # Setup logging again to avoid logger from being overwritten during setting up the pipeline components
    setup_logging()

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi: RTVIProcessor):
        logger.info("Pipecat client ready.")
        await rtvi.set_bot_ready()
        # Kick off the conversation.
        try:
            await task.queue_frames([user_context_aggregator.get_context_frame()])
        except Exception as e:
            logger.error(f"Error queuing context frame: {e}")

    @ws_transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Pipecat Client connected from {client.remote_address}")
        # Reset RTVI state for new connection
        rtvi._client_ready = False
        rtvi._bot_ready = False

    @ws_transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Pipecat Client disconnected from {client.remote_address}")
        # Finalize audio logger session if enabled
        if audio_logger:
            audio_logger.finalize_session()
            logger.info("Audio logger session finalized")
        # Don't cancel the task immediately - let it handle the disconnection gracefully
        # The task will continue running and can accept new connections
        # Only send an EndTaskFrame to clean up the current session
        if task_running:
            try:
                await task.queue_frames([EndTaskFrame()])
            except Exception as e:
                # Don't log warnings for normal connection closures
                if "ConnectionClosedOK" not in str(e) and "1005" not in str(e):
                    logger.warning(f"Error sending EndTaskFrame: {e}")
                else:
                    logger.info(f"Normal connection closure: {e}")

    @ws_transport.event_handler("on_session_timeout")
    async def on_session_timeout(transport, client):
        logger.info(f"Session timeout for {client.remote_address}")
        # Don't cancel the task - keep server running indefinitely
        logger.info("Session timeout occurred but keeping server running")
        # Note: With session_timeout=None, this handler should never be called

    logger.info("Starting pipeline runner...")

    try:
        runner = PipelineRunner()
        # Run the task until shutdown is requested
        await asyncio.wait_for(runner.run(task), timeout=None)  # No timeout - run indefinitely
    except asyncio.TimeoutError:
        logger.info("Pipeline runner timeout (should not happen with no timeout)")
    except Exception as e:
        logger.error(f"Pipeline runner error: {e}")
        task_running = False
    finally:
        # Finalize audio logger on shutdown
        if audio_logger:
            audio_logger.finalize_session()
            logger.info("Audio logger session finalized on shutdown")
        logger.info("Pipeline runner stopped")


if __name__ == "__main__":
    asyncio.run(run_bot_websocket_server())
