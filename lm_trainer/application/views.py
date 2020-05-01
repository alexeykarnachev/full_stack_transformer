from typing import Optional, Sequence

from lm_trainer.application.schemas import (
    GeneratedTexts,
    SeedText,
    TextGeneratorAppParams
)
from lm_trainer.text_generator.text_generator import (
    TextGenerator,
    TextGeneratorParams
)
from lm_trainer.tokenizers import Tokenizer


def register_generated_texts_view(
        app,
        generator: TextGenerator,
        tokenizer: Tokenizer,
        ignored_token_ids: Optional[Sequence[int]]
):
    @app.post("/generated_texts/", response_model=GeneratedTexts)
    def generated_texts(
            seed_text: SeedText,
            text_generator_app_params: TextGeneratorAppParams
    ) -> GeneratedTexts:
        seed_token_ids = _tokenize_seed_text(seed_text.text, tokenizer)

        text_generator_params = TextGeneratorParams(
            seed_token_ids=seed_token_ids,
            ignored_token_ids=ignored_token_ids,
            **text_generator_app_params.dict()
        )

        texts = generator(text_generator_params)

        return GeneratedTexts(texts=texts)

    return generated_texts


def _tokenize_seed_text(text, tokenizer):
    seed_token_ids = [tokenizer.get_bos_token_id()]
    if len(text):
        seed_token_ids.append(tokenizer.encode(text))

    return seed_token_ids


def _decode_token_ids(token_ids, tokenizer):
    generated = tokenizer.decode_batch(token_ids)
    generated = [tokenizer.postprocess(seq) for seq in generated]
    return generated
