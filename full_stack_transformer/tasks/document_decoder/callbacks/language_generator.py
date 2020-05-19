from full_stack_transformer.tasks.common.callbacks.language_generator import LanguageGeneratorCallback
from full_stack_transformer.tasks.document_decoder.text_input import DocumentInput
from full_stack_transformer.utilities.experiment import Workspace


class LanguageGeneratorDocumentCallback(LanguageGeneratorCallback):
    def __init__(
            self,
            experiment_workspace: Workspace
    ):
        text_input = DocumentInput(body='', meta=None)
        super().__init__(
            experiment_workspace=experiment_workspace,
            default_text_input=text_input
        )
