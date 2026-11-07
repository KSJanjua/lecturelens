### exp1_transcription

| lecture | language | engine | reference | model | wer | seconds | latin_script_share | audio_minutes |
|---|---|---|---|---|---|---|---|---|
| L1 | en | api | human captions | whisper-large-v3 | 0.0304 |  | 0.984 | 18.7 |
| L1 | en | local | human captions | faster-whisper small | 0.0091 | 631.9 | 0.984 | 18.7 |
| L2 | en | api | human captions | whisper-large-v3 | 0.0386 |  | 0.991 | 20.5 |
| L2 | en | local | human captions | faster-whisper small | 0.0386 | 848.3 | 0.993 | 20.5 |
| L3 | en | api | human captions | whisper-large-v3 | 0.0286 |  | 0.99 | 12.8 |
| L3 | en | local | human captions | faster-whisper small | 0.0286 | 481.0 | 0.995 | 12.8 |
| L4 | en | api | human captions | whisper-large-v3 | 0.0729 |  | 0.982 | 11.9 |
| L4 | en | local | human captions | faster-whisper small | 0.0654 | 537.5 | 0.984 | 11.9 |
| L5 | en | api | human captions | whisper-large-v3 | 0.0495 |  | 0.994 | 13.2 |
| L5 | en | local | human captions | faster-whisper small | 0.0536 | 427.3 | 0.995 | 13.2 |
| L6 | hi | api | youtube-auto (silver) | whisper-large-v3 | 0.6221 |  | 0.355 | 20.5 |
| L6 | hi | local | youtube-auto (silver) | faster-whisper small | 0.8342 | 15017.1 | 0.166 | 20.5 |

### exp2_prompting

| lecture | strategy | rouge1 | rouge2 | rougeL | bertscore_f1 | faithfulness | claims | unjudged | human_rating | tokens | api_calls | seconds | generator | judge |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L3 | zero_shot | 0.5336 | 0.2333 | 0.2375 | 0.7973 | 0.86 | 50 | 0 |  | 9846 | 5 | 0.0 | groq/openai/gpt-oss-120b | groq/qwen/qwen3.8-27b |
| L3 | few_shot | 0.4734 | 0.2157 | 0.2527 | 0.8049 | 0.943 | 88 | 0 |  | 9983 | 5 | 0.0 | groq/openai/gpt-oss-120b | groq/qwen/qwen3.8-27b |
| L3 | cot | 0.5249 | 0.2234 | 0.2152 | 0.8037 | 0.97 | 66 | 0 |  | 12185 | 5 | 0.0 | groq/openai/gpt-oss-120b | groq/qwen/qwen3.8-27b |
| L3 | tot | 0.5105 | 0.2177 | 0.2264 | 0.7974 | 0.936 | 47 | 0 |  | 56268 | 29 | 0.0 | groq/openai/gpt-oss-120b | groq/qwen/qwen3.8-27b |
| L1 | zero_shot | 0.4092 | 0.1632 | 0.1747 | 0.7561 | 0.943 | 88 | 0 |  | 12714 | 6 | 0.0 | groq/openai/gpt-oss-120b | groq/qwen/qwen3.8-27b |
| L1 | few_shot | 0.3606 | 0.1508 | 0.1749 | 0.7556 | 0.955 | 66 | 0 |  | 12654 | 6 | 0.0 | groq/openai/gpt-oss-120b | groq/qwen/qwen3.8-27b |
| L1 | cot | 0.3967 | 0.1552 | 0.1687 | 0.7513 | 0.953 | 64 | 0 |  | 13839 | 6 | 0.0 | groq/openai/gpt-oss-120b | groq/qwen/qwen3.8-27b |

### exp2_summary

| strategy | strategy_label | lectures | rouge1 | rouge2 | rougeL | bertscore_f1 | faithfulness | human_rating | tokens | generator | judge |
|---|---|---|---|---|---|---|---|---|---|---|---|
| zero_shot | Zero-shot | 1 | 0.5336 | 0.2333 | 0.2375 | 0.7973 | 0.86 |  | 9846 | groq/openai/gpt-oss-120b | groq/qwen/qwen3.8-27b |
| few_shot | Few-shot | 1 | 0.4734 | 0.2157 | 0.2527 | 0.8049 | 0.943 |  | 9983 | groq/openai/gpt-oss-120b | groq/qwen/qwen3.8-27b |
| cot | Chain-of-Thought | 1 | 0.5249 | 0.2234 | 0.2152 | 0.8037 | 0.97 |  | 12185 | groq/openai/gpt-oss-120b | groq/qwen/qwen3.8-27b |
| tot | Tree-of-Thought (3x2) | 1 | 0.5105 | 0.2177 | 0.2264 | 0.7974 | 0.936 |  | 56268 | groq/openai/gpt-oss-120b | groq/qwen/qwen3.8-27b |

### exp3_sampling

| lecture | temperature | top_p | distinct2 | faithfulness | unjudged | tokens | generator | judge |
|---|---|---|---|---|---|---|---|---|
| L3 | 0.2 | 0.9 | 0.802 | 0.955 | 0 | 5711 | groq/openai/gpt-oss-20b | groq/qwen/qwen3.8-27b |
| L3 | 0.5 | 0.9 | 0.7598 | 0.878 | 0 | 6078 | groq/openai/gpt-oss-20b | groq/qwen/qwen3.8-27b |
| L3 | 0.8 | 0.9 | 0.8511 | 0.931 | 0 | 5731 | groq/openai/gpt-oss-20b | groq/qwen/qwen3.8-27b |
| L3 | 1.1 | 0.9 | 0.8218 | 0.967 | 0 | 5816 | groq/openai/gpt-oss-20b | groq/qwen/qwen3.8-27b |
| L3 | 0.2 | 1.0 | 0.7986 | 0.969 | 0 | 5885 | groq/openai/gpt-oss-20b | groq/qwen/qwen3.8-27b |
| L3 | 0.5 | 1.0 | 0.7931 | 0.892 | 0 | 5967 | groq/openai/gpt-oss-20b | groq/qwen/qwen3.8-27b |
| L3 | 0.8 | 1.0 | 0.787 | 0.857 | 0 | 5915 | groq/openai/gpt-oss-20b | groq/qwen/qwen3.8-27b |
| L3 | 1.1 | 1.0 | 0.8066 | 1.0 | 0 | 5984 | groq/openai/gpt-oss-20b | groq/qwen/qwen3.8-27b |

### exp4_chunking

| lecture | chunking | chunks | mean_chunk_tokens | rag_accuracy | citation_accuracy | hit_at_1 | mrr | hit_at_4 | rag_questions |
|---|---|---|---|---|---|---|---|---|---|
| L1 | fixed-1000 | 5 | 878 |  |  | 1.0 | 1.0 | 1.0 | 6 |
| L1 | fixed-matched | 6 | 732 |  |  | 1.0 | 1.0 | 1.0 | 6 |
| L1 | semantic | 5 | 878 |  |  | 1.0 | 1.0 | 1.0 | 6 |
| L2 | fixed-1000 | 5 | 972 |  |  | 0.75 | 0.8 | 0.75 | 4 |
| L2 | fixed-matched | 8 | 608 |  |  | 0.5 | 0.619 | 0.75 | 4 |
| L2 | semantic | 7 | 694 |  |  | 1.0 | 1.0 | 1.0 | 4 |
| L3 | fixed-1000 | 4 | 746 |  |  | 0.0 | 0.444 | 1.0 | 3 |
| L3 | fixed-matched | 5 | 597 |  |  | 1.0 | 1.0 | 1.0 | 3 |
| L3 | semantic | 4 | 746 |  |  | 1.0 | 1.0 | 1.0 | 3 |
| L4 | fixed-1000 | 4 | 760 |  |  | 0.75 | 0.875 | 1.0 | 5 |
| L4 | fixed-matched | 6 | 506 |  |  | 1.0 | 1.0 | 1.0 | 5 |
| L4 | semantic | 5 | 608 |  |  | 0.75 | 0.875 | 1.0 | 5 |
| L5 | fixed-1000 | 4 | 790 |  |  | 0.75 | 0.812 | 1.0 | 4 |
| L5 | fixed-matched | 7 | 452 |  |  | 0.75 | 0.875 | 1.0 | 4 |
| L5 | semantic | 6 | 527 |  |  | 0.75 | 0.875 | 1.0 | 4 |

### exp5_personalisation

| lecture | profile | fk_grade | words | bullets | definitions | tokens | generator | judge |
|---|---|---|---|---|---|---|---|---|
| L3 | beginner | 15.69 | 876 | 25 | 7 | 7618 | groq/openai/gpt-oss-20b | gemini/gemini-3.5-flash-lite |
| L3 | exam revision | 15.06 | 716 | 26 | 3 | 7272 | groq/openai/gpt-oss-20b | gemini/gemini-3.5-flash-lite |
| L6 | beginner | 12.19 | 2061 | 51 | 25 | 16973 | groq/openai/gpt-oss-20b | gemini/gemini-3.5-flash-lite |
| L6 | exam revision | 10.91 | 1645 | 44 | 19 | 15832 | groq/openai/gpt-oss-20b | gemini/gemini-3.5-flash-lite |
| L4 | beginner | 14.11 | 1238 | 43 | 7 | 9542 | groq/openai/gpt-oss-20b | gemini/gemini-3.5-flash-lite |
| L4 | exam revision | 14.87 | 996 | 34 | 8 | 8891 | groq/openai/gpt-oss-20b | gemini/gemini-3.5-flash-lite |
| L5 | beginner | 13.35 | 1127 | 35 | 19 | 10589 | groq/openai/gpt-oss-20b | gemini/gemini-3.5-flash-lite |
| L5 | exam revision | 12.78 | 887 | 31 | 14 | 9794 | groq/openai/gpt-oss-20b | gemini/gemini-3.5-flash-lite |

### exp6_quiz

| lecture | variant | generated | passed_first_try | answerability_rate | dropped | tokens | generator | judge |
|---|---|---|---|---|---|---|---|---|
| L3 | no CoT | 10 | 10 | 1.0 | 0 | 12260 | groq/openai/gpt-oss-20b | gemini/gemini-3.5-flash-lite |
| L3 | CoT | 10 | 10 | 1.0 | 0 | 8250 | groq/openai/gpt-oss-20b | gemini/gemini-3.5-flash-lite |
| L4 | no CoT | 10 | 10 | 1.0 | 0 | 8632 | groq/openai/gpt-oss-20b | gemini/gemini-3.5-flash-lite |
| L4 | CoT | 10 | 10 | 1.0 | 0 | 12271 | groq/openai/gpt-oss-20b | gemini/gemini-3.5-flash-lite |
| L5 | no CoT | 10 | 10 | 1.0 | 0 | 8929 | groq/openai/gpt-oss-20b | gemini/gemini-3.5-flash-lite |
| L5 | CoT | 10 | 10 | 1.0 | 0 | 10258 | groq/openai/gpt-oss-20b | gemini/gemini-3.5-flash-lite |
| L6 | no CoT | 10 | 10 | 1.0 | 0 | 11234 | groq/openai/gpt-oss-20b | gemini/gemini-3.5-flash-lite |
| L6 | CoT | 10 | 10 | 1.0 | 0 | 11992 | groq/openai/gpt-oss-20b | gemini/gemini-3.5-flash-lite |
| L1 | no CoT | 10 | 9 | 0.9 | 0 | 17630 | groq/openai/gpt-oss-20b | gemini/gemini-3.5-flash-lite |
| L1 | CoT | 10 | 10 | 1.0 | 0 | 11625 | groq/openai/gpt-oss-20b | gemini/gemini-3.5-flash-lite |
| L2 | no CoT | 10 | 10 | 1.0 | 0 | 11713 | groq/openai/gpt-oss-20b | gemini/gemini-3.5-flash-lite |
| L2 | CoT | 10 | 10 | 1.0 | 0 | 12721 | groq/openai/gpt-oss-20b | gemini/gemini-3.5-flash-lite |
