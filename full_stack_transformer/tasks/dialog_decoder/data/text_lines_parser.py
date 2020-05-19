import json
from typing import List

from full_stack_transformer.core.data.text_lines_parsers import TextLinesParser
from full_stack_transformer.tasks.dialog_decoder.text_input import DialogInput


class DialogLinesParser(TextLinesParser):
    def __init__(self):
        pass

    def parse(self, text: str) -> List[DialogInput]:

        body = json.loads(text)
        utts = body['utterances']
        tags = body.get('tags')

        inps = []
        for idx in (0, 1):
            pers = body.get(f'persona_{idx}') or None
            if pers:
                inp = DialogInput(
                    utterances=utts, tags=tags, persona=pers, persona_idx=idx
                )
                inps.append(inp)

        if len(inps) == 0:
            inp = DialogInput(
                utterances=utts, tags=tags, persona=None, persona_idx=None
            )
            inps.append(inp)

        return inps
