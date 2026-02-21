#!/bin/bash

conda create -n nemo-voice python=3.12
conda activate nemo-voice
conda install nvidia::cuda-toolkit
pip install torch==2.9.1 torchaudio torchvision --index-url https://download.pytorch.org/whl/cu130
pip install https://github.com/vllm-project/vllm/releases/download/v0.15.1/vllm-0.15.1+cu130-cp38-abi3-manylinux_2_35_aarch64.whl
pip install python_weather kokoro websockets silero-vad
pip install "nemo-toolkit[asr,tts]==2.6.2" --no-deps
pip install "pipecat-ai[silero,openai,runner,local-smart-turn-v3,webrtc]==0.0.98" --no-deps