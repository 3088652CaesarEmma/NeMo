import os
from pathlib import Path

import nltk
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


def get_llm(model_name="Qwen/Qwen3-4B-Instruct-2507"):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["HF_HOME"] = "/home/vbataev/hf_models"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    llm = LLM(
        model_name,
        **{
            "dtype": "auto",
            "seed": 42,
            "gpu_memory_utilization": 0.5,
            "max_model_len": 8192,
        },
    )
    return llm


def get_uttid(record):
    return Path(record["audio_filepath"]).stem


def translate_manifest(
    manifest,
    llm,
    sampling_params,
    prompt_template,
    target_language: str,
    source_language: str = "English",
    num_keep_sentences=5,
    text_key="pred_text",
) -> list[str]:
    translations = []
    for record in tqdm(manifest):
        text = record[text_key]
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
    return translations


def main():
    llm = get_llm()

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

    manifest_path = Path(
        "/home/vbataev/code/worktrees/nemo/iwslt26/_checks/asr/en-096-096/baseline/streaming_greedy_dev-long.jsonl"
    )

    manifest = read_manifest(manifest_path)
    # source_language = "English"
    # target_language = "German"
    # target_language = "Russian"
    hyp_translations_de = translate_manifest(
        manifest,
        llm=llm,
        sampling_params=sampling_params,
        prompt_template=EuroLLMTranslatorPromptTemplate(),
        target_language="German",
    )
    hyp_translations_it = translate_manifest(
        manifest,
        llm=llm,
        sampling_params=sampling_params,
        prompt_template=EuroLLMTranslatorPromptTemplate(),
        target_language="Italian",
    )

    # prompt_template = EuroLLMTranslatorPromptTemplate()
    for i, record in enumerate(manifest):
        record["hyp_translation_de"] = hyp_translations_de[i]
        record["hyp_translation_it"] = hyp_translations_it[i]

    all_ref_manifest = read_manifest("/data/iwslt26/mcif/manifests/manifest_en_all.json")
    ref_translations_de = []
    ref_translations_it = []
    for record, record_with_ref in zip(manifest, all_ref_manifest):
        assert get_uttid(record) == get_uttid(record_with_ref)
        record["ref_translation_de"] = record_with_ref["ref_translation_de"]
        ref_translations_de.append(record_with_ref["ref_translation_de"])
        record["ref_translation_it"] = record_with_ref["ref_translation_it"]
        ref_translations_it.append(record_with_ref["ref_translation_it"])

    write_manifest(manifest_path.parent / f"{manifest_path.stem}__hyp-en-it.jsonl", manifest)


if __name__ == "__main__":
    main()
