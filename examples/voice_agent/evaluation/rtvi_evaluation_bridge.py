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

"""
RTVI Evaluation Bridge

Connects two voice agents via WebSocket and provides:
- Bidirectional audio routing
- Response latency measurement
- Dynamic system prompt updates via RTVI actions
- Conversation monitoring and metrics
"""

import asyncio
import json
import struct
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import websockets
from loguru import logger
from pipecat.processors.frameworks.rtvi import (
    RTVIBotStartedSpeakingMessage,
    RTVIBotStoppedSpeakingMessage,
    RTVIBotTranscriptionMessage,
    RTVITextMessageData,
)

# RTVI message type constants - automatically adapts to pipecat changes
RTVI_BOT_STOPPED_SPEAKING = RTVIBotStoppedSpeakingMessage().type
RTVI_BOT_STARTED_SPEAKING = RTVIBotStartedSpeakingMessage().type
RTVI_BOT_TRANSCRIPTION = RTVIBotTranscriptionMessage(data=RTVITextMessageData(text="")).type


@dataclass
class ResponseLatency:
    """Single response latency measurement"""

    evaluator_stop_time: float  # When evaluator stopped speaking
    target_start_time: float  # When target started responding
    latency_ms: float  # Response latency in milliseconds
    evaluator_transcript: str = ""
    target_transcript: str = ""


@dataclass
class SegmentEntry:
    """Entry for segLST format (segment list with timing)"""

    start_time: float  # Start time in seconds
    end_time: float  # End time in seconds
    speaker: str  # "evaluator" or "target"
    transcript: str  # Text content


@dataclass
class EvaluationMetrics:
    """Metrics collected during evaluation"""

    turns: list = field(default_factory=list)
    latencies: List[ResponseLatency] = field(default_factory=list)
    start_time: datetime = None
    end_time: datetime = None

    # Audio timing state
    evaluator_last_audio_time: Optional[float] = None
    target_last_audio_time: Optional[float] = None
    waiting_for_target_response: bool = False
    last_evaluator_transcript: str = ""

    # Transcript accumulation (segments arrive incrementally)
    evaluator_current_transcript: str = ""
    target_current_transcript: str = ""

    # Audio recording (for stereo WAV output)
    evaluator_audio_chunks: List[bytes] = field(default_factory=list)
    target_audio_chunks: List[bytes] = field(default_factory=list)
    audio_sample_rate: int = 16000  # Default sample rate
    audio_start_timestamp: Optional[float] = None

    # Segment tracking for segLST output
    segments: List[SegmentEntry] = field(default_factory=list)
    current_segment_start: Optional[float] = None

    def get_latency_stats(self):
        """Calculate latency statistics"""
        if not self.latencies:
            return {
                "count": 0,
                "mean_ms": 0,
                "median_ms": 0,
                "p95_ms": 0,
                "min_ms": 0,
                "max_ms": 0,
            }

        latencies_sorted = sorted([l.latency_ms for l in self.latencies])
        count = len(latencies_sorted)

        return {
            "count": count,
            "mean_ms": sum(latencies_sorted) / count,
            "median_ms": latencies_sorted[count // 2],
            "p95_ms": latencies_sorted[int(count * 0.95)] if count > 0 else 0,
            "min_ms": latencies_sorted[0],
            "max_ms": latencies_sorted[-1],
        }


class RTVIEvaluationBridge:
    """
    Evaluation bridge that connects two voice agents via WebSocket
    and provides control through RTVI actions.

    Key features:
    - Routes audio bidirectionally between agents
    - Monitors transcriptions and metrics
    - Measures response latency by tracking audio frames
    - Can send RTVI control messages to update prompts
    - Works with distributed agents
    """

    def __init__(
        self,
        evaluator_url: str,
        target_url: str,
        log_file: Optional[str] = None,
        audio_file: Optional[str] = None,
        evaluator_sample_rate: int = 16000,
        target_sample_rate: int = 16000,
        output_sample_rate: int = 16000,
    ):
        self.evaluator_url = evaluator_url
        self.target_url = target_url
        self.log_file = log_file
        self.audio_file = audio_file
        self.evaluator_sample_rate = evaluator_sample_rate
        self.target_sample_rate = target_sample_rate
        self.output_sample_rate = output_sample_rate

        self.evaluator_ws = None
        self.target_ws = None

        self.metrics = EvaluationMetrics()
        self.metrics.audio_sample_rate = output_sample_rate

        # Track RTVI state
        self.evaluator_ready = False
        self.target_ready = False

        # Initialize log file
        if self.log_file:
            with open(self.log_file, "w") as f:
                f.write("RTVI Evaluation Bridge - Conversation Log\n")
                f.write("=" * 80 + "\n")
                f.write(f"Start Time: {datetime.now().isoformat()}\n")
                f.write("=" * 80 + "\n\n")

    async def connect(self):
        """Connect to both agents"""
        logger.info(f"Connecting to evaluator at {self.evaluator_url}")
        self.evaluator_ws = await websockets.connect(self.evaluator_url)

        logger.info(f"Connecting to target at {self.target_url}")
        self.target_ws = await websockets.connect(self.target_url)

        self.metrics.start_time = datetime.now()

        # Send client-ready handshake to both
        await self._send_client_ready(self.target_ws)
        await self._send_client_ready(self.evaluator_ws)

        logger.info("Both agents connected and ready")

    async def _send_client_ready(self, ws):
        """Send RTVI client-ready message"""
        client_ready_msg = {
            "type": "rtvi",
            "label": "rtvi-ai",
            "protocol_version": "0.2.0",
            "data": {"message_type": "client-ready"},
        }
        await ws.send(json.dumps(client_ready_msg))

    async def update_evaluator_prompt(self, new_prompt: str, auto_reset: bool = True):
        """
        Send RTVI action to update evaluator's system prompt.

        Args:
            new_prompt: New system prompt text
            auto_reset: If True, also sends reset action after updating prompt
        """
        logger.info(f"Updating evaluator prompt: {new_prompt[:100]}...")

        # Send update_system_prompt action
        action_msg = {
            "type": "rtvi",
            "label": "rtvi-ai",
            "data": {
                "message_type": "action-run",
                "id": f"update_prompt_{datetime.now().timestamp()}",
                "service": "context",
                "action": "update_system_prompt",
                "arguments": [{"name": "prompt", "value": new_prompt}],
            },
        }

        await self.evaluator_ws.send(json.dumps(action_msg))

        # Wait for update response
        response = await self._wait_for_action_response(self.evaluator_ws)
        if not response:
            logger.error("Failed to update evaluator prompt")
            return False

        logger.info("Evaluator prompt updated successfully")

        # Send reset action if requested (though update handler already resets)
        if auto_reset:
            logger.info("Sending additional reset action to evaluator...")
            await self._send_reset_action(self.evaluator_ws, "evaluator")

        return True

    async def update_target_prompt(self, new_prompt: str, auto_reset: bool = True):
        """
        Send RTVI action to update target's system prompt.

        Args:
            new_prompt: New system prompt text
            auto_reset: If True, also sends reset action after updating prompt
        """
        logger.info(f"Updating target prompt: {new_prompt[:100]}...")

        # Send update_system_prompt action
        action_msg = {
            "type": "rtvi",
            "label": "rtvi-ai",
            "data": {
                "message_type": "action-run",
                "id": f"update_prompt_{datetime.now().timestamp()}",
                "service": "context",
                "action": "update_system_prompt",
                "arguments": [{"name": "prompt", "value": new_prompt}],
            },
        }

        await self.target_ws.send(json.dumps(action_msg))

        # Wait for update response
        response = await self._wait_for_action_response(self.target_ws)
        if not response:
            logger.error("Failed to update target prompt")
            return False

        logger.info("Target prompt updated successfully")

        # Send reset action if requested (though update handler already resets)
        if auto_reset:
            logger.info("Sending additional reset action to target...")
            await self._send_reset_action(self.target_ws, "target")

        return True

    async def _send_reset_action(self, ws, agent_name: str):
        """
        Send RTVI reset action to clear conversation history.

        Args:
            ws: WebSocket connection
            agent_name: Name of agent (for logging)
        """
        reset_msg = {
            "type": "rtvi",
            "label": "rtvi-ai",
            "data": {
                "message_type": "action-run",
                "id": f"reset_{datetime.now().timestamp()}",
                "service": "context",
                "action": "reset",
                "arguments": [],
            },
        }

        await ws.send(json.dumps(reset_msg))

        # Wait for reset response
        response = await self._wait_for_action_response(ws)
        if response:
            logger.info(f"{agent_name.capitalize()} conversation reset successfully")
        else:
            logger.warning(f"Failed to reset {agent_name} conversation")

    async def reset_conversation(self):
        """
        Reset both agents' conversation history.
        Useful to clear context between evaluation scenarios.
        """
        logger.info("Resetting both agents...")
        await self._send_reset_action(self.evaluator_ws, "evaluator")
        await self._send_reset_action(self.target_ws, "target")

        # Reset latency tracking state
        self.metrics.evaluator_last_audio_time = None
        self.metrics.target_last_audio_time = None
        self.metrics.waiting_for_target_response = False
        self.metrics.last_evaluator_transcript = ""

        # Clear accumulated transcript segments
        self.metrics.evaluator_current_transcript = ""
        self.metrics.target_current_transcript = ""

        logger.info("Both agents reset complete")

    async def _wait_for_action_response(self, ws, timeout=5.0):
        """Wait for RTVI action response"""
        try:
            start_time = asyncio.get_event_loop().time()
            while asyncio.get_event_loop().time() - start_time < timeout:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.5)

                    if isinstance(msg, str):
                        data = json.loads(msg)

                        if data.get("data", {}).get("message_type") == "action-response":
                            result = data.get("data", {}).get("result", {})
                            return result.get("success", False) or result is True
                except asyncio.TimeoutError:
                    continue

            logger.warning("Timeout waiting for action response")
            return False
        except Exception as e:
            logger.error(f"Error waiting for response: {e}")
            return False

    async def route_audio(self, duration: int = 300):
        """Route audio between agents and monitor conversation"""

        async def forward_evaluator_to_target():
            """Forward evaluator audio/messages to target"""
            try:
                async for message in self.evaluator_ws:
                    # Track timing and monitor
                    await self._monitor_evaluator_message(message)
                    # Forward to target
                    await self.target_ws.send(message)
            except websockets.exceptions.ConnectionClosed:
                logger.info("Evaluator connection closed")
            except Exception as e:
                logger.error(f"Error in evaluator->target forwarding: {e}")

        async def forward_target_to_evaluator():
            """Forward target audio/messages to evaluator"""
            try:
                async for message in self.target_ws:
                    # Track timing and monitor
                    await self._monitor_target_message(message)
                    # Forward to evaluator
                    await self.evaluator_ws.send(message)
            except websockets.exceptions.ConnectionClosed:
                logger.info("Target connection closed")
            except Exception as e:
                logger.error(f"Error in target->evaluator forwarding: {e}")

        # Run with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(forward_evaluator_to_target(), forward_target_to_evaluator()), timeout=duration
            )
        except asyncio.TimeoutError:
            logger.info(f"Evaluation duration of {duration}s reached")
        except Exception as e:
            logger.error(f"Error during audio routing: {e}")

    async def _monitor_evaluator_message(self, message):
        """
        Monitor evaluator messages for timing and transcripts.
        This tracks when evaluator sends audio and stops speaking.
        """
        timestamp = asyncio.get_event_loop().time()

        # Track binary audio frames from evaluator
        if isinstance(message, bytes):
            self.metrics.evaluator_last_audio_time = timestamp

            # Record audio if audio_file is specified
            if self.audio_file:
                # Initialize audio start timestamp on first audio
                if self.metrics.audio_start_timestamp is None:
                    self.metrics.audio_start_timestamp = timestamp
                    self.metrics.current_segment_start = timestamp

                # Save audio chunk
                self.metrics.evaluator_audio_chunks.append(message)

            return

        # Parse JSON messages
        try:
            data = json.loads(message)
            message_type = data.get("data", {}).get("message_type", "")

            # Track evaluator transcription segments (accumulate)
            if message_type == RTVI_BOT_TRANSCRIPTION:
                text = data.get("data", {}).get("text", "")
                if text:
                    # Accumulate text segments (they arrive incrementally)
                    self.metrics.evaluator_current_transcript += text
                    logger.debug(f"[EVALUATOR SEGMENT] {text}")

            # Track when evaluator bot stops speaking (finalize turn)
            elif message_type == RTVI_BOT_STOPPED_SPEAKING:
                self.metrics.evaluator_last_audio_time = timestamp
                self.metrics.waiting_for_target_response = True
                logger.debug(f"[TIMING] Evaluator stopped speaking at {timestamp:.3f}")

                # Finalize the turn with accumulated transcript
                if self.metrics.evaluator_current_transcript:
                    complete_text = self.metrics.evaluator_current_transcript.strip()
                    self.metrics.last_evaluator_transcript = complete_text
                    logger.info(f"[EVALUATOR] {complete_text}")

                    turn_data = {
                        "timestamp": datetime.now().isoformat(),
                        "role": "evaluator",
                        "text": complete_text,
                    }
                    self.metrics.turns.append(turn_data)

                    if self.log_file:
                        with open(self.log_file, "a") as f:
                            f.write(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] EVALUATOR: {complete_text}\n")

                    # Create segment entry for segLST
                    if self.audio_file and self.metrics.current_segment_start is not None:
                        segment_start = self.metrics.current_segment_start - (self.metrics.audio_start_timestamp or 0)
                        segment_end = timestamp - (self.metrics.audio_start_timestamp or 0)
                        segment = SegmentEntry(
                            start_time=segment_start,
                            end_time=segment_end,
                            speaker="evaluator",
                            transcript=complete_text,
                        )
                        self.metrics.segments.append(segment)
                        self.metrics.current_segment_start = None

                    # Clear accumulated text for next turn
                    self.metrics.evaluator_current_transcript = ""

        except json.JSONDecodeError:
            pass
        except Exception as e:
            logger.error(f"Error monitoring evaluator message: {e}")

    async def _monitor_target_message(self, message):
        """
        Monitor target messages for timing and transcripts.
        This tracks when target starts responding (first audio received).
        """
        timestamp = asyncio.get_event_loop().time()

        # Track binary audio frames from target - this is the response!
        if isinstance(message, bytes):
            # If we're waiting for target response and this is the first audio
            if self.metrics.waiting_for_target_response and self.metrics.evaluator_last_audio_time:
                latency_ms = (timestamp - self.metrics.evaluator_last_audio_time) * 1000

                # Create latency measurement
                latency = ResponseLatency(
                    evaluator_stop_time=self.metrics.evaluator_last_audio_time,
                    target_start_time=timestamp,
                    latency_ms=latency_ms,
                    evaluator_transcript=self.metrics.last_evaluator_transcript,
                )

                self.metrics.latencies.append(latency)
                self.metrics.waiting_for_target_response = False

                logger.info(f"[LATENCY] Response latency: {latency_ms:.1f}ms")

                if self.log_file:
                    with open(self.log_file, "a") as f:
                        f.write(f"  → Response latency: {latency_ms:.1f}ms\n")

                # Track segment start for target when it starts responding
                if self.audio_file and self.metrics.current_segment_start is None:
                    self.metrics.current_segment_start = timestamp

            self.metrics.target_last_audio_time = timestamp

            # Record audio if audio_file is specified
            if self.audio_file:
                # Initialize audio start timestamp on first audio
                if self.metrics.audio_start_timestamp is None:
                    self.metrics.audio_start_timestamp = timestamp
                    self.metrics.current_segment_start = timestamp

                # Save audio chunk
                self.metrics.target_audio_chunks.append(message)

            return

        # Parse JSON messages
        try:
            data = json.loads(message)
            message_type = data.get("data", {}).get("message_type", "")

            # Track when target bot starts speaking
            if message_type == RTVI_BOT_STARTED_SPEAKING:
                if self.metrics.waiting_for_target_response and self.metrics.evaluator_last_audio_time:
                    latency_ms = (timestamp - self.metrics.evaluator_last_audio_time) * 1000

                    logger.debug(f"[TIMING] Target started speaking at {timestamp:.3f} (latency: {latency_ms:.1f}ms)")

            # Track target transcription segments (accumulate)
            elif message_type == RTVI_BOT_TRANSCRIPTION:
                text = data.get("data", {}).get("text", "")
                if text:
                    # Accumulate text segments (they arrive incrementally)
                    self.metrics.target_current_transcript += text
                    logger.debug(f"[TARGET SEGMENT] {text}")

            # Track when target bot stops speaking (finalize turn)
            elif message_type == RTVI_BOT_STOPPED_SPEAKING:
                logger.debug(f"[TIMING] Target stopped speaking at {timestamp:.3f}")

                # Finalize the turn with accumulated transcript
                if self.metrics.target_current_transcript:
                    complete_text = self.metrics.target_current_transcript.strip()
                    logger.info(f"[TARGET] {complete_text}")

                    # Update the last latency measurement with complete target transcript
                    if self.metrics.latencies and not self.metrics.latencies[-1].target_transcript:
                        self.metrics.latencies[-1].target_transcript = complete_text

                    turn_data = {
                        "timestamp": datetime.now().isoformat(),
                        "role": "target",
                        "text": complete_text,
                    }
                    self.metrics.turns.append(turn_data)

                    if self.log_file:
                        with open(self.log_file, "a") as f:
                            f.write(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] TARGET: {complete_text}\n")

                    # Create segment entry for segLST
                    if self.audio_file and self.metrics.current_segment_start is not None:
                        segment_start = self.metrics.current_segment_start - (self.metrics.audio_start_timestamp or 0)
                        segment_end = timestamp - (self.metrics.audio_start_timestamp or 0)
                        segment = SegmentEntry(
                            start_time=segment_start, end_time=segment_end, speaker="target", transcript=complete_text
                        )
                        self.metrics.segments.append(segment)
                        self.metrics.current_segment_start = None

                    # Clear accumulated text for next turn
                    self.metrics.target_current_transcript = ""

        except json.JSONDecodeError:
            pass
        except Exception as e:
            logger.error(f"Error monitoring target message: {e}")

    def _resample_audio(self, audio_chunks: List[bytes], from_rate: int, to_rate: int) -> np.ndarray:
        """
        Resample audio chunks from one sample rate to another.

        Args:
            audio_chunks: List of audio byte chunks (16-bit PCM)
            from_rate: Source sample rate
            to_rate: Target sample rate

        Returns:
            Resampled audio as numpy array
        """
        if not audio_chunks:
            return np.array([], dtype=np.int16)

        # Concatenate all chunks
        audio_bytes = b''.join(audio_chunks)

        # Convert bytes to int16 array
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

        # If sample rates match, no resampling needed
        if from_rate == to_rate:
            return audio_array

        # Simple linear resampling
        duration = len(audio_array) / from_rate
        target_length = int(duration * to_rate)

        # Use numpy interp for resampling
        x_old = np.linspace(0, duration, len(audio_array))
        x_new = np.linspace(0, duration, target_length)
        resampled = np.interp(x_new, x_old, audio_array)

        return resampled.astype(np.int16)

    async def _save_audio_and_seglst(self):
        """Save stereo audio file and segLST transcript file."""
        try:
            logger.info(f"Saving audio to {self.audio_file}...")

            # Resample both channels to output sample rate
            evaluator_audio = self._resample_audio(
                self.metrics.evaluator_audio_chunks, self.evaluator_sample_rate, self.output_sample_rate
            )
            target_audio = self._resample_audio(
                self.metrics.target_audio_chunks, self.target_sample_rate, self.output_sample_rate
            )

            # Make both arrays the same length (pad shorter one with zeros)
            max_length = max(len(evaluator_audio), len(target_audio))
            if len(evaluator_audio) < max_length:
                evaluator_audio = np.pad(evaluator_audio, (0, max_length - len(evaluator_audio)))
            if len(target_audio) < max_length:
                target_audio = np.pad(target_audio, (0, max_length - len(target_audio)))

            # Create stereo array (evaluator=left, target=right)
            stereo_audio = np.empty((max_length, 2), dtype=np.int16)
            stereo_audio[:, 0] = evaluator_audio  # Left channel
            stereo_audio[:, 1] = target_audio  # Right channel

            # Save as WAV file
            with wave.open(self.audio_file, 'wb') as wav_file:
                wav_file.setnchannels(2)  # Stereo
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.output_sample_rate)
                wav_file.writeframes(stereo_audio.tobytes())

            logger.info(f"Audio saved: {self.audio_file}")
            logger.info(f"  Channels: 2 (evaluator=left, target=right)")
            logger.info(f"  Sample rate: {self.output_sample_rate} Hz")
            logger.info(f"  Duration: {max_length / self.output_sample_rate:.2f}s")

            # Save segLST file
            seglst_file = Path(self.audio_file).with_suffix('.seglst')
            with open(seglst_file, 'w') as f:
                f.write("# segLST format: start_time end_time speaker transcript\n")
                f.write(f"# Audio file: {Path(self.audio_file).name}\n")
                f.write(f"# Sample rate: {self.output_sample_rate} Hz\n")
                f.write("#\n")

                # Write segments sorted by start time
                sorted_segments = sorted(self.metrics.segments, key=lambda s: s.start_time)
                for seg in sorted_segments:
                    # Format: start end speaker text
                    f.write(f"{seg.start_time:.3f} {seg.end_time:.3f} {seg.speaker} {seg.transcript}\n")

            logger.info(f"segLST saved: {seglst_file}")
            logger.info(f"  Total segments: {len(self.metrics.segments)}")

        except Exception as e:
            logger.error(f"Error saving audio/segLST: {e}")
            import traceback

            traceback.print_exc()

    async def disconnect(self):
        """Disconnect from both agents"""
        self.metrics.end_time = datetime.now()

        if self.evaluator_ws:
            await self.evaluator_ws.close()
        if self.target_ws:
            await self.target_ws.close()

        logger.info("Disconnected from both agents")

        # Save audio and segLST if configured
        if self.audio_file:
            await self._save_audio_and_seglst()

        # Log final statistics
        latency_stats = self.metrics.get_latency_stats()
        if latency_stats['count'] > 0:
            logger.info(f"\nFinal Latency Statistics:")
            logger.info(f"  Measurements: {latency_stats['count']}")
            logger.info(f"  Mean: {latency_stats['mean_ms']:.1f}ms")
            logger.info(f"  Median: {latency_stats['median_ms']:.1f}ms")
            logger.info(f"  P95: {latency_stats['p95_ms']:.1f}ms")
            logger.info(f"  Min: {latency_stats['min_ms']:.1f}ms")
            logger.info(f"  Max: {latency_stats['max_ms']:.1f}ms")

    def get_metrics(self):
        """Get evaluation metrics"""
        duration = 0
        if self.metrics.start_time and self.metrics.end_time:
            duration = (self.metrics.end_time - self.metrics.start_time).total_seconds()

        latency_stats = self.metrics.get_latency_stats()

        return {
            "total_turns": len(self.metrics.turns),
            "duration_seconds": duration,
            "turns": self.metrics.turns,
            "latency_stats": latency_stats,
            "latencies": [
                {
                    "evaluator_transcript": l.evaluator_transcript,
                    "target_transcript": l.target_transcript,
                    "latency_ms": l.latency_ms,
                }
                for l in self.metrics.latencies
            ],
        }
