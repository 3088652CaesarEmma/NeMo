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

import inspect
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logging(log_file: str = "bot_server.log", log_level: str = "DEBUG", rotation: str = "1 day"):
    # Configure loguru to output to both console and file
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
    )

    logger.add(log_file, rotation=rotation, level=log_level)


class FileLogger:
    """Simple file+stdout logger with caller location tracking."""

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file

    def _get_caller_location(self) -> str:
        """Return file:function:line of the caller, skipping frames inside FileLogger."""
        logger_methods = {"log", "info", "error", "warning", "debug", "__call__", "_get_caller_location"}
        for frame_info in inspect.stack():
            if frame_info.function not in logger_methods:
                path = Path(frame_info.filename).resolve()
                return f"{path.name}:{frame_info.function}:{frame_info.lineno}"
        return "unknown"

    def log(self, message: str, include_caller: bool = True):
        if include_caller:
            message = f"{self._get_caller_location()} | {message}"

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"{timestamp} | {message}"
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(message + "\n")
        print(message, flush=True)

    def __call__(self, message: str, include_caller: bool = True):
        self.log(message, include_caller=include_caller)

    def info(self, message: str, include_caller: bool = True):
        self.log(f"[INFO]: {message}", include_caller=include_caller)

    def error(self, message: str, include_caller: bool = True):
        self.log(f"[ERROR]: {message}", include_caller=include_caller)

    def warning(self, message: str, include_caller: bool = True):
        self.log(f"[WARNING]: {message}", include_caller=include_caller)

    def debug(self, message: str, include_caller: bool = True):
        self.log(f"[DEBUG]: {message}", include_caller=include_caller)
