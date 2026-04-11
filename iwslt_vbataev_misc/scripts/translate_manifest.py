import os
from pathlib import Path

import nltk
import regex
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams

from nemo.collections.asr.inference.nmt.prompts import (
    EuroLLMTranslatorPromptTemplate,
    PromptTemplate,
    Qwen3TranslatorPromptTemplate,
)
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest

# nltk.download('punkt')
# nltk.download('punkt_tab')


def main():
    source_language = "English"
    target_language = "German"
    # target_language = "Russian"

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["HF_HOME"] = "/home/vbataev/hf_models"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    llm = LLM(
        model_name,
        **{
            "dtype": "auto",
            "seed": 42,
            "gpu_memory_utilization": 0.5,
            "max_model_len": 8192,
        },
    )

    sampling_params = SamplingParams(
        **{
            "max_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.8,  # The cumulative probability threshold for nucleus sampling
            "top_k": 20,  # The number of top tokens to sample from
            "min_p": 0,  # The minimum probability threshold for sampling
            "presence_penalty": 1.5,  # The presence penalty for sampling
            "seed": 42,  # The seed to initialize the random number generator for sampling
            "stop": ["<|im_end|>", "\u200d"],
        }
    )

    prompt_template = EuroLLMTranslatorPromptTemplate()

    base_path = Path("/home/vbataev/code/worktrees/nemo/iwslt26/_checks/asr/en-096-096/baseline/")

    manifest = read_manifest(base_path / "streaming_greedy_dev-long.jsonl")
    translations = []

    num_keep_sentences = 5
    for record in tqdm(manifest):
        text = record["pred_text"]
        sentences = nltk.sent_tokenize(text)
        per_sentence_translations = []
        for i, sentence in enumerate(tqdm(sentences, leave=False)):
            llm_input = prompt_template.format(
                source_language,
                target_language,
                src_prefix=sentence,
                tgt_prefix="",
                src_context=" ".join(sentences[max(i - num_keep_sentences, 0) : i]),
                tgt_context=" ".join(per_sentence_translations[max(i - num_keep_sentences, 0) : i]),
            )
            llm_output = llm.generate([llm_input], sampling_params, use_tqdm=False)
            output_text = llm_output[0].outputs[0].text
            output_text = prompt_template.extract(output_text).strip()
            per_sentence_translations.append(output_text)
        translation = " ".join(per_sentence_translations)
        translations.append(translation)
        record["pred_text_de"] = translation
    write_manifest(base_path / "streaming_greedy_dev-long_with_translation.jsonl", manifest)
    with open(base_path / "streaming_greedy_dev-long_asr-to-de.txt", "w", encoding="utf-8") as f:
        for translation in translations:
            print(translation, file=f)


if __name__ == "__main__":
    main()
