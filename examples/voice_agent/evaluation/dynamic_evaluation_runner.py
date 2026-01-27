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
Dynamic Voice Agent Evaluation Runner

Runs evaluation scenarios with dynamic system prompt updates.
Each scenario can specify different prompts for evaluator and target agents.
"""

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from loguru import logger
from rtvi_evaluation_bridge import RTVIEvaluationBridge


async def run_dynamic_evaluation(
    evaluator_url: str,
    target_url: str,
    output_dir: str,
    scenarios: list[dict],
    duration_per_scenario: int = 60,
    pause_between_scenarios: int = 5,
    record_audio: bool = True,
    output_sample_rate: int = 16000,
):
    """
    Run evaluation with dynamic scenario switching and latency measurement.

    Args:
        evaluator_url: WebSocket URL of evaluator agent
        target_url: WebSocket URL of target agent
        output_dir: Output directory for results
        scenarios: List of scenarios, each with:
            - name: Scenario name
            - evaluator_prompt: Evaluator system prompt
            - target_prompt: Optional target system prompt
            - duration: Optional duration override
        duration_per_scenario: Default duration per scenario (seconds)
        pause_between_scenarios: Seconds to pause between scenarios
        record_audio: Whether to record audio (default: True)
        output_sample_rate: Output sample rate for audio (default: 16000)
    """

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create audio directory if recording
    if record_audio:
        audio_dir = os.path.join(output_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)

    bridge = RTVIEvaluationBridge(
        evaluator_url=evaluator_url,
        target_url=target_url,
        log_file=os.path.join(output_dir, f"conversation_{timestamp}.log"),
        audio_file=None,  # Will be set per scenario
        output_sample_rate=output_sample_rate,
    )

    await bridge.connect()

    all_results = []

    for idx, scenario in enumerate(scenarios):
        logger.info(f"\n{'='*80}")
        logger.info(f"Starting Scenario {idx+1}/{len(scenarios)}: {scenario['name']}")
        logger.info(f"{'='*80}\n")

        # Reset metrics for new scenario
        bridge.metrics.latencies = []
        bridge.metrics.turns = []
        bridge.metrics.segments = []

        # Reset audio buffers
        bridge.metrics.evaluator_audio_chunks = []
        bridge.metrics.target_audio_chunks = []
        bridge.metrics.audio_start_timestamp = None
        bridge.metrics.current_segment_start = None

        # Set audio file for this scenario
        if record_audio:
            scenario_name_safe = scenario['name'].replace(' ', '_').replace('/', '_')
            audio_file = os.path.join(audio_dir, f"{timestamp}_{idx+1:02d}_{scenario_name_safe}.wav")
            bridge.audio_file = audio_file
            logger.info(f"Recording audio to: {audio_file}")
        else:
            bridge.audio_file = None

        # Update prompts (auto_reset=True will reset after updating)
        await bridge.update_evaluator_prompt(scenario["evaluator_prompt"], auto_reset=False)  # Handler already resets

        if "target_prompt" in scenario:
            await bridge.update_target_prompt(scenario["target_prompt"], auto_reset=False)  # Handler already resets
        elif idx > 0:
            # Reset target to clear history from previous scenario
            await bridge._send_reset_action(bridge.target_ws, "target")

        # Wait for agents to stabilize after reset
        logger.info("Waiting for agents to stabilize after prompt update...")
        await asyncio.sleep(3)

        # Run scenario
        duration = scenario.get("duration", duration_per_scenario)
        logger.info(f"Running scenario for {duration} seconds...")

        scenario_start = datetime.now()
        await bridge.route_audio(duration=duration)
        scenario_end = datetime.now()

        # Collect metrics for this scenario
        metrics = bridge.get_metrics()
        metrics["scenario_name"] = scenario["name"]
        metrics["scenario_duration"] = (scenario_end - scenario_start).total_seconds()
        all_results.append(metrics)

        # Log scenario summary
        latency_stats = metrics["latency_stats"]
        logger.info(f"\n{'='*80}")
        logger.info(f"Scenario '{scenario['name']}' Complete")
        logger.info(f"{'='*80}")
        logger.info(f"  Total turns: {metrics['total_turns']}")
        logger.info(f"  Duration: {metrics['scenario_duration']:.1f}s")
        logger.info(f"  Latency measurements: {latency_stats['count']}")
        if latency_stats['count'] > 0:
            logger.info(f"  Mean latency: {latency_stats['mean_ms']:.1f}ms")
            logger.info(f"  Median latency: {latency_stats['median_ms']:.1f}ms")
            logger.info(f"  P95 latency: {latency_stats['p95_ms']:.1f}ms")

        # Pause between scenarios
        if idx < len(scenarios) - 1:
            logger.info(f"\nPausing {pause_between_scenarios} seconds before next scenario...")
            await asyncio.sleep(pause_between_scenarios)

    await bridge.disconnect()

    # Save detailed results
    results_file = os.path.join(output_dir, f"results_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Save CSV with latency details
    latency_csv_file = os.path.join(output_dir, f"latencies_{timestamp}.csv")
    with open(latency_csv_file, "w") as f:
        f.write("Scenario,Evaluator_Transcript,Target_Transcript,Latency_ms\n")
        for result in all_results:
            scenario_name = result["scenario_name"]
            for latency in result["latencies"]:
                eval_text = latency["evaluator_transcript"].replace('"', '""')
                target_text = latency["target_transcript"].replace('"', '""')
                f.write(f'"{scenario_name}","{eval_text}","{target_text}",{latency["latency_ms"]:.1f}\n')

    # Save summary
    summary_file = os.path.join(output_dir, f"summary_{timestamp}.txt")
    with open(summary_file, "w") as f:
        f.write("EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        total_turns = sum(r["total_turns"] for r in all_results)
        total_duration = sum(r["scenario_duration"] for r in all_results)

        f.write(f"Total Scenarios: {len(scenarios)}\n")
        f.write(f"Total Duration: {total_duration:.1f}s\n")
        f.write(f"Total Turns: {total_turns}\n\n")

        f.write("Per-Scenario Results:\n")
        f.write("-" * 80 + "\n")
        for result in all_results:
            stats = result["latency_stats"]
            f.write(f"\n{result['scenario_name']}:\n")
            f.write(f"  Turns: {result['total_turns']}\n")
            f.write(f"  Duration: {result['scenario_duration']:.1f}s\n")
            if result['scenario_duration'] > 0:
                f.write(f"  Turns/min: {result['total_turns'] / (result['scenario_duration'] / 60):.1f}\n")
            f.write(f"  Latency Measurements: {stats['count']}\n")
            if stats['count'] > 0:
                f.write(f"    Mean: {stats['mean_ms']:.1f}ms\n")
                f.write(f"    Median: {stats['median_ms']:.1f}ms\n")
                f.write(f"    P95: {stats['p95_ms']:.1f}ms\n")
                f.write(f"    Min: {stats['min_ms']:.1f}ms\n")
                f.write(f"    Max: {stats['max_ms']:.1f}ms\n")

        # Overall latency statistics
        all_latencies = []
        for result in all_results:
            all_latencies.extend([l["latency_ms"] for l in result["latencies"]])

        if all_latencies:
            all_latencies.sort()
            count = len(all_latencies)
            f.write(f"\n\nOverall Latency Statistics:\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Total Measurements: {count}\n")
            f.write(f"  Mean: {sum(all_latencies) / count:.1f}ms\n")
            f.write(f"  Median: {all_latencies[count // 2]:.1f}ms\n")
            f.write(f"  P95: {all_latencies[int(count * 0.95)]:.1f}ms\n")
            f.write(f"  Min: {all_latencies[0]:.1f}ms\n")
            f.write(f"  Max: {all_latencies[-1]:.1f}ms\n")

    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluation Complete!")
    logger.info(f"{'='*80}")
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"Latencies saved to: {latency_csv_file}")
    logger.info(f"Summary saved to: {summary_file}")
    logger.info(f"Conversation log: {bridge.log_file}")
    if record_audio:
        logger.info(f"Audio files saved to: {audio_dir}/")
        logger.info(f"  Format: Stereo WAV (evaluator=left, target=right)")
        logger.info(f"  Sample rate: {output_sample_rate} Hz")
        logger.info(f"  segLST files: {audio_dir}/*.seglst")
    logger.info(f"\nTotal: {len(scenarios)} scenarios, {total_turns} turns, {total_duration:.1f}s")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run voice agent evaluation with dynamic scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default scenarios
  python dynamic_evaluation_runner.py \\
      --evaluator-url ws://localhost:8765 \\
      --target-url ws://localhost:8766

  # Run with custom scenarios file
  python dynamic_evaluation_runner.py \\
      --evaluator-url ws://localhost:8765 \\
      --target-url ws://localhost:8766 \\
      --scenarios-file scenarios/customer_service.json \\
      --output-dir ./results/cs_eval

  # Run with custom duration
  python dynamic_evaluation_runner.py \\
      --evaluator-url ws://localhost:8765 \\
      --target-url ws://localhost:8766 \\
      --duration 120 \\
      --pause 10
        """,
    )
    parser.add_argument(
        "--evaluator-url",
        default="ws://localhost:8765",
        help="WebSocket URL of evaluator agent (default: ws://localhost:8765)",
    )
    parser.add_argument(
        "--target-url",
        default="ws://localhost:8766",
        help="WebSocket URL of target agent (default: ws://localhost:8766)",
    )
    parser.add_argument(
        "--output-dir", default="./eval_results", help="Output directory for results (default: ./eval_results)"
    )
    parser.add_argument("--scenarios-file", help="JSON file with scenarios (see scenarios/ directory for examples)")
    parser.add_argument(
        "--duration", type=int, default=60, help="Default duration per scenario in seconds (default: 60)"
    )
    parser.add_argument("--pause", type=int, default=5, help="Pause between scenarios in seconds (default: 5)")
    parser.add_argument(
        "--no-audio", action="store_true", help="Disable audio recording (default: audio recording is enabled)"
    )
    parser.add_argument(
        "--output-sample-rate", type=int, default=16000, help="Output sample rate for recorded audio (default: 16000)"
    )

    args = parser.parse_args()

    # Default scenarios
    scenarios = [
        {
            "name": "Friendly Conversation",
            "evaluator_prompt": """You are a friendly user testing a voice assistant.
Ask casual questions and engage in pleasant conversation.
Start by greeting the assistant and asking about its capabilities.""",
            "duration": 60,
        },
        {
            "name": "Challenging Questions",
            "evaluator_prompt": """You are testing a voice assistant with difficult questions.
Ask complex, multi-part questions and test edge cases.
Start with a challenging question about a technical topic.""",
            "duration": 60,
        },
        {
            "name": "Rapid Interaction",
            "evaluator_prompt": """You are testing how well the assistant handles quick back-and-forth.
Ask short questions and wait for answers. Keep responses brief.
Start with a simple question and build from there.""",
            "duration": 60,
        },
    ]

    # Load scenarios from file if provided
    if args.scenarios_file:
        scenarios_path = Path(args.scenarios_file)
        if not scenarios_path.exists():
            logger.error(f"Scenarios file not found: {args.scenarios_file}")
            return 1

        with open(scenarios_path) as f:
            scenarios = json.load(f)
        logger.info(f"Loaded {len(scenarios)} scenarios from {args.scenarios_file}")
    else:
        logger.info(f"Using {len(scenarios)} default scenarios")

    # Run evaluation
    try:
        asyncio.run(
            run_dynamic_evaluation(
                evaluator_url=args.evaluator_url,
                target_url=args.target_url,
                output_dir=args.output_dir,
                scenarios=scenarios,
                duration_per_scenario=args.duration,
                pause_between_scenarios=args.pause,
                record_audio=not args.no_audio,
                output_sample_rate=args.output_sample_rate,
            )
        )
        return 0
    except KeyboardInterrupt:
        logger.info("\nEvaluation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
