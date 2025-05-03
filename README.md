# hutton-lm

A prototype language model interface for generating geological models!

Uses [Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) + [GemPy](https://www.gempy.org/)!

## Examples

<img width="887" alt="Screenshot 2025-05-03 at 3 48 09 PM" src="https://github.com/user-attachments/assets/b0fcad6f-910f-4cb0-97a8-57e81d425940" />

## To run

To run, use the following command:

```
export LLAMA_API_KEY=<your-key-here>

python data-to-3d-model-input.py \
    --input-mode llm \
    --prompt-type default \
    --llm-output-dir input-data/llm-generated
```
