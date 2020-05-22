from full_stack_transformer.tasks.common.callbacks.language_generator import LanguageGeneratorCallback
from full_stack_transformer.tasks.dialog_decoder.text_input import DialogInput
from full_stack_transformer.utilities.experiment import Workspace


class LanguageGeneratorDialogCallback(LanguageGeneratorCallback):
    def __init__(
            self,
            experiment_workspace: Workspace
    ):
        text_input = DialogInput(
            utterances=[''],
            persona='',
            persona_idx=0,
            tags=''
        )
        super().__init__(
            experiment_workspace=experiment_workspace,
            default_text_input=text_input
        )
