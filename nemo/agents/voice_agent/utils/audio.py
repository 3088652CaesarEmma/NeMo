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

import asyncio
import time
from typing import List, Optional, Tuple, Union

import librosa
import numpy as np
import soxr
from loguru import logger

STREAM_TIMEOUT_SECS = 0.2


class NoiseGenerator:
    """
    A class that generates noise audio by reading provided noise audio files.
    """

    def __init__(
        self,
        noise_audio_files: Union[List[str], str],
        sample_rate: int,
        max_duration: Optional[float] = None,
        random_offset: bool = True,
    ):
        """
        Args:
            noise_audio_files: List of noise audio files to load or a single noise audio file.
            sample_rate: Sample rate of the output audio chunks.
            max_duration: Maximum duration of each noise audio file to load.
            random_offset: Whether to randomize the offset of the noise audio if it's longer than the maximum duration.
        """
        if not isinstance(noise_audio_files, list):
            noise_audio_files = str(noise_audio_files).split(',')

        self.noise_audio_files = noise_audio_files
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.random_offset = random_offset
        self.noise_audio_data = self.load_audio_files()
        self.current_position = 0  # Track current position in samples

    def load_audio_files(self) -> np.ndarray:
        """
        Load the noise audio files.
        """
        logger.info(f"Loading {len(self.noise_audio_files)} noise audio files...")
        noise_audio_data = []
        for noise_audio_file in self.noise_audio_files:
            audio_duration = librosa.get_duration(path=noise_audio_file)
            if self.random_offset and self.max_duration is not None and audio_duration > self.max_duration:
                offset = np.random.uniform(0, audio_duration - self.max_duration)
            else:
                offset = 0
            noise_audio_segment, _ = librosa.load(
                noise_audio_file, sr=self.sample_rate, duration=self.max_duration, offset=offset
            )
            noise_audio_data.append(noise_audio_segment)

        # concatenate the noise audio data into a single array
        return np.concatenate(noise_audio_data)

    def get_noise_chunk(self, chunk_size_in_seconds: float) -> np.ndarray:
        """
        Get the next noise audio segment of chunk size chunk_size_in_seconds, and return the chunk.
        If the noise audio data is less than the chunk size, restart from the beginning.

        Args:
            chunk_size_in_seconds: Duration of the noise chunk to return in seconds.

        Returns:
            np.ndarray: Noise audio chunk of the requested duration.
        """
        # Calculate chunk size in samples
        chunk_size_in_samples = int(chunk_size_in_seconds * self.sample_rate)

        # Get total length of noise audio data
        total_samples = len(self.noise_audio_data)

        # If the chunk size is larger than the total available noise, repeat the noise
        if chunk_size_in_samples > total_samples:
            # Calculate how many times we need to repeat
            num_repeats = (chunk_size_in_samples // total_samples) + 1
            noise_chunk = np.tile(self.noise_audio_data, num_repeats)[:chunk_size_in_samples]
            # Reset position to handle the wraparound
            self.current_position = chunk_size_in_samples % total_samples
            return noise_chunk

        # Check if we have enough samples from current position
        end_position = self.current_position + chunk_size_in_samples

        if end_position <= total_samples:
            # We have enough samples without wrapping around
            noise_chunk = self.noise_audio_data[self.current_position : end_position]
            self.current_position = end_position % total_samples  # Wrap to 0 if we hit the end
        else:
            # We need to wrap around to the beginning
            samples_from_end = total_samples - self.current_position
            samples_from_start = chunk_size_in_samples - samples_from_end

            # Concatenate the end and beginning portions
            noise_chunk = np.concatenate(
                [self.noise_audio_data[self.current_position :], self.noise_audio_data[:samples_from_start]]
            )
            self.current_position = samples_from_start

        return noise_chunk.copy().clip(-1.0, 1.0)

    def get_noise_chunk_bytes(self, chunk_size_in_seconds: float) -> bytes:
        """
        Get the next noise audio segment of chunk size chunk_size_in_seconds, and return the chunk as Int16 bytes.
        """
        noise_chunk = self.get_noise_chunk(chunk_size_in_seconds)
        return (noise_chunk * 32767.0).astype(np.int16).tobytes()


class SOXRAudioResampler:
    """
    An audio resampler that uses the SoX resampler library. It's stateless and will return the result immediately.
    """

    def __init__(self, in_sample_rate: int, out_sample_rate: int, quality: str = "VHQ", *args, **kwargs):
        """Initialize the SoX audio resampler.

        Args:
            in_sample_rate: The sample rate of the input audio.
            out_sample_rate: The sample rate of the output audio.
            quality: The quality of the resampling.
            **kwargs: Additional keyword arguments (currently unused).
        """
        self.quality = quality
        self.in_sample_rate = in_sample_rate
        self.out_sample_rate = out_sample_rate

    def resample(self, audio: bytes) -> bytes:
        """Resample audio data using SoX resampler library.

        Args:
            audio: Input audio data as raw bytes (16-bit signed integers).

        Returns:
            Resampled audio data as raw bytes (16-bit signed integers).
        """
        if self.in_sample_rate == self.out_sample_rate:
            return audio
        audio_data = np.frombuffer(audio, dtype=np.int16)
        resampled_audio = soxr.resample(audio_data, self.in_sample_rate, self.out_sample_rate, quality=self.quality)
        result = resampled_audio.astype(np.int16).tobytes()
        return result


class SOXRAudioStreamResampler:
    """
    A class that resamples an audio stream using the SoX resampler library.
    """

    def __init__(self, in_sample_rate: int, out_sample_rate: int, quality: str = "VHQ", *args, **kwargs):
        self.in_sample_rate = in_sample_rate
        self.out_sample_rate = out_sample_rate
        self.quality = quality
        self.resampler = soxr.ResampleStream(
            in_sample_rate, out_sample_rate, quality=quality, num_channels=1, dtype="int16"
        )
        self._last_resample_time = None

    def _should_flush(self):
        """
        Check if the resampler should be flushed.
        """
        if self._last_resample_time is None:
            return False
        return time.time() - self._last_resample_time > STREAM_TIMEOUT_SECS

    def reset(self):
        """
        Reset the resampler.
        """
        self._last_resample_time = None
        self.resampler.clear()

    def resample(self, audio: bytes):
        """
        Resample an audio chunk using the SoX resampler library.
        Args:
            audio: The audio chunk to resample.
        Returns:
            The resampled audio chunk.
        """
        is_last = self._should_flush()
        audio_data = np.frombuffer(audio, dtype=np.int16)
        resampled_audio = self.resampler.resample_chunk(audio_data, last=is_last)
        self._last_resample_time = time.time()
        if is_last:
            self.reset()
        result = resampled_audio.astype(np.int16).tobytes()
        return result


class AudioStream:
    """
    A class that simulates a realtime audio stream. It caches the input audio chunks
    and resamples them to the output sample rate. Each time its get() function is called,
    it returns the next chunk of audio at the output sample rate. If the audio cache doesn't
    have enough audio to fill the output chunk, it will append silence to the output chunk.

    The class will be used in an asyncio context, where one thread is putting audio chunks
    into the cache and another thread is getting audio chunks from the cache.
    """

    def __init__(
        self,
        chunk_size_in_seconds: float,
        input_sample_rate: int,
        output_sample_rate: int,
        stream_resampler: bool = True,
        tag: str = "",
        min_buffer_chunks: int = 5,
        drain_threshold: int = 5,
        min_sustain_chunks: int = 1,
        noise_files: Union[List[str], str] = None,
        min_snr_db: float = 0.0,
        max_snr_db: float = 30.0,
        max_gain_db: float = 300.0,
        max_noise_duration: Optional[float] = 600.0,
    ):
        self.chunk_size_in_seconds = chunk_size_in_seconds
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.stream_resampler = stream_resampler
        self.output_chunk_bytes = int(self.output_sample_rate * self.chunk_size_in_seconds) * 2  # 16-bit audio
        self.tag = tag
        self.min_buffer_chunks = min_buffer_chunks
        self._buffer_ready = False
        self.noise_files = noise_files
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.max_gain_db = max_gain_db
        if self.noise_files is not None:
            self.noise_generator = NoiseGenerator(
                self.noise_files, self.output_sample_rate, max_duration=max_noise_duration
            )
        else:
            self.noise_generator = None
        # Initialize the appropriate resampler
        if self.stream_resampler:
            self.resampler = SOXRAudioStreamResampler(input_sample_rate, output_sample_rate, quality="VHQ")
        else:
            self.resampler = SOXRAudioResampler(input_sample_rate, output_sample_rate, quality="VHQ")

        # Use asyncio.Queue for async/await compatibility
        self.audio_cache = asyncio.Queue()

        # Buffer for partial chunks
        self.output_buffer = b''

        self._buffer_empty_count = 0  # Track consecutive empty returns
        self.drain_threshold = drain_threshold  # Only reset ready after 5 consecutive underflows (~80ms of silence)
        self.min_sustain_chunks = min_sustain_chunks
        self._next_send_time = 0

    async def put(self, audio_chunk: bytes):
        """
        Put an audio chunk into the audio cache after resampling.

        Args:
            audio_chunk: Input audio chunk at input_sample_rate
        """
        # Resample the audio chunk to output sample rate
        await self.audio_cache.put(audio_chunk)
        audio_len_in_seconds = len(audio_chunk) / 2 / self.input_sample_rate
        logger.debug(
            f"[{self.tag}] Put {len(audio_chunk)} bytes ({audio_len_in_seconds:.4f} seconds) into AudioStream"
        )

    def resample(self, audio_chunk: bytes) -> bytes:
        """
        Resample an audio chunk from input sample rate to output sample rate.

        Args:
            audio_chunk: Raw audio bytes (16-bit signed integers)

        Returns:
            Resampled audio bytes (16-bit signed integers)
        """
        if self.input_sample_rate == self.output_sample_rate:
            return audio_chunk

        return self.resampler.resample(audio_chunk)

    def _augment_with_noise(self, audio_chunk: bytes, noise_chunk: bytes, target_length_bytes: int) -> bytes:
        """
        Augment audio with noise based on random SNR sampling.

        This method mixes audio with noise according to a randomly sampled SNR value
        from the range [min_snr_db, max_snr_db]. The noise is scaled to achieve the
        target SNR, with a maximum gain limit to prevent excessive amplification.

        Args:
            audio_chunk: Original audio bytes (16-bit signed integers)
            noise_chunk: Noise audio bytes (16-bit signed integers)
            target_length_bytes: Target output length in bytes

        Returns:
            Mixed audio with noise, exactly target_length_bytes long
        """
        # Step 1: Prepare arrays
        audio_samples = len(audio_chunk) // 2  # 16-bit = 2 bytes per sample
        noise_samples_needed = target_length_bytes // 2

        # Step 2: Prepare noise (trim or pad to target length)
        noise_int16 = np.frombuffer(noise_chunk, dtype=np.int16).astype(np.float32)
        if len(noise_int16) > noise_samples_needed:
            noise_int16 = noise_int16[:noise_samples_needed]
        elif len(noise_int16) < noise_samples_needed:
            # Pad noise with zeros if too short
            padding = np.zeros(noise_samples_needed - len(noise_int16), dtype=np.float32)
            noise_int16 = np.concatenate([noise_int16, padding])

        # Step 3: Sample random SNR
        target_snr_db = np.random.uniform(self.min_snr_db, self.max_snr_db)

        # Step 4: Calculate noise scale factor
        if audio_samples > 0:
            # We have signal, calculate SNR-based scale
            audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
            signal_power = np.mean(audio_int16**2)
            noise_power = np.mean(noise_int16**2)

            if noise_power > 0 and signal_power > 0:
                # SNR_dB = 10 * log10(P_signal / P_noise_target)
                # P_noise_target = P_signal / 10^(SNR_dB/10)
                # scale = sqrt(P_noise_target / P_noise_original)
                target_noise_power = signal_power / (10 ** (target_snr_db / 10))
                noise_scale = np.sqrt(target_noise_power / noise_power)
            else:
                noise_scale = 1.0
        else:
            # No signal (pure padding), use max gain
            noise_scale = 10 ** (self.max_gain_db / 20)

        # Step 5: Apply max gain limit
        max_scale = 10 ** (self.max_gain_db / 20)
        noise_scale = min(noise_scale, max_scale)

        # Step 6: Scale noise
        scaled_noise = noise_int16 * noise_scale

        # Step 7: Mix audio + noise
        output = np.zeros(noise_samples_needed, dtype=np.float32)
        if audio_samples > 0:
            # Mix signal + noise for overlapping region
            audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
            output[:audio_samples] = audio_int16 + scaled_noise[:audio_samples]
        # Use only noise for padding region
        output[audio_samples:] = scaled_noise[audio_samples:]

        # Step 8: Clip to int16 range and convert to bytes
        output = np.clip(output, -32768, 32767).astype(np.int16)
        return output.tobytes()

    def get_output_chunk(self, audio_chunk: bytes, noise_chunk: Optional[bytes] = None) -> bytes:
        """
        Pad audio chunk with silence/noise if shorter than expected output chunk size.

        If noise_chunk is provided and noise_generator is available, the audio will be
        augmented with noise based on SNR. Otherwise, it pads with silence (zeros).

        Args:
            audio_chunk: Audio bytes (16-bit signed integers)
            noise_chunk: Optional noise bytes for augmentation

        Returns:
            Audio chunk padded to output_chunk_bytes
        """
        current_length = len(audio_chunk)
        if current_length >= self.output_chunk_bytes:
            # Trim to exact size
            audio_chunk = audio_chunk[: self.output_chunk_bytes]
        else:
            # Pad with silence (zeros)
            padding_bytes = self.output_chunk_bytes - current_length
            audio_chunk = audio_chunk + (b'\x00' * padding_bytes)

        if noise_chunk is not None:
            # Use noise augmentation
            return self._augment_with_noise(audio_chunk, noise_chunk, self.output_chunk_bytes)
        else:
            return audio_chunk

    @property
    def current_buffer_size(self) -> int:
        """
        Get the current size of the buffer.
        """
        return len(self.output_buffer) // self.output_chunk_bytes

    def _is_buffer_full(self) -> bool:
        """
        Check if the buffer is full.
        """
        return self.current_buffer_size >= self.min_buffer_chunks

    async def _send_audio_sleep(self):
        """Simulate audio device timing by sleeping between audio chunks."""
        # Simulate a clock.
        current_time = time.monotonic()
        sleep_duration = max(0, self._next_send_time - current_time)
        await asyncio.sleep(sleep_duration)
        if sleep_duration == 0:
            self._next_send_time = time.monotonic() + self.chunk_size_in_seconds
        else:
            self._next_send_time += self.chunk_size_in_seconds

    async def get_nowait(self) -> Tuple[bytes, bool]:
        """
        Get the next output chunk of audio, immediately padding with silence if no audio is available.
        """
        return await self.get_wait(no_wait=True)

    async def get_wait(self, timeout: float = None, no_wait: bool = False) -> Tuple[bytes, bool]:
        """
        Get the next output chunk of audio, WAITING for audio to be available.

        Unlike get(), this method will block and wait for audio to arrive rather than
        immediately padding with silence. This prevents gaps in audio when packets
        arrive in bursts (common in WebSocket/network scenarios).

        Use this for continuous audio streaming where you want smooth audio without
        artificial gaps.

        Args:
            timeout: Maximum time to wait in seconds (None = no wait)
            no_wait: If True, only tries to read the audio cache once, and returns silence
                    immediately if no audio is available.
        Returns:
            Tuple[audio_chunk, has_speech]: Tuple containing the audio chunk bytes and a
                    boolean indicating if there's speech in the chunk
        """
        start_time = time.time()
        if no_wait:
            timeout = None
        while True:
            try:
                # Calculate remaining time budget BEFORE waiting
                if timeout is not None:
                    elapsed = time.time() - start_time
                    remaining_timeout = timeout - elapsed
                    if remaining_timeout <= 0:
                        break  # Out of time budget
                else:
                    remaining_timeout = None
                if remaining_timeout is not None:
                    chunk = await asyncio.wait_for(self.audio_cache.get(), timeout=remaining_timeout)
                else:
                    chunk = self.audio_cache.get_nowait()
                chunk = self.resample(chunk)
                self.output_buffer += chunk
                logger.debug(
                    f"[{self.tag}] Added {len(chunk)} bytes ({len(chunk) / 2 / self.output_sample_rate:.4f} seconds) to buffer, current buffer size: {self.current_buffer_size}"
                )
                if self._is_buffer_full() or no_wait:
                    break
            except (asyncio.TimeoutError, asyncio.QueueEmpty):
                break

        if self._is_buffer_full():
            self._buffer_ready = True

        # Check if buffer too low to sustain
        if self._buffer_ready and self.current_buffer_size < self.min_sustain_chunks:
            # Only reset if we've been low for a while
            self._buffer_empty_count += 1
            if self._buffer_empty_count > self.drain_threshold:
                self._buffer_ready = False
                logger.warning(
                    f"[{self.tag}] Buffer sustained low, resetting (empty count: {self._buffer_empty_count})"
                )
        else:
            self._buffer_empty_count = 0

        # get noise chunk if needed
        if self.noise_generator is not None:
            noise_chunk = self.noise_generator.get_noise_chunk_bytes(self.chunk_size_in_seconds)
        else:
            noise_chunk = b''

        if not self._buffer_ready:
            logger.debug(
                f"[{self.tag}] Buffer not ready ({self.current_buffer_size}/{self.min_buffer_chunks} chunks), sending silence"
            )
            # Return the output chunk and a boolean indicating if there's speech in the chunk
            return self.get_output_chunk(b'', noise_chunk), False

        logger.debug(
            f"[{self.tag}] Buffer ready ({self.current_buffer_size}/{self.min_buffer_chunks} chunks), sending audio"
        )
        output_chunk = self.output_buffer
        # If we have more than needed, split it
        if len(output_chunk) > self.output_chunk_bytes:
            output_audio_chunk = output_chunk[: self.output_chunk_bytes]
            self.output_buffer = output_chunk[self.output_chunk_bytes :]
        elif len(output_chunk) == self.output_chunk_bytes:
            # Exactly the right amount
            self.output_buffer = b''
            output_audio_chunk = output_chunk
        else:
            # Buffer has partial chunk, pad with noise/silence
            self._buffer_empty_count += 1
            # Only reset ready after 5 consecutive underflows (~80ms of silence)
            if self._buffer_empty_count > self.drain_threshold:
                self._buffer_ready = False
                logger.warning(f"[{self.tag}] Buffer drained, resetting (empty count: {self._buffer_empty_count})")
            logger.debug(f"[{self.tag}] Buffer partial, padding (empty count: {self._buffer_empty_count})")
            self.output_buffer = b''
            output_audio_chunk = output_chunk

        # Return the output chunk and a boolean indicating if there's speech in the chunk
        return self.get_output_chunk(output_audio_chunk, noise_chunk), True
