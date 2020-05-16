from collections import namedtuple


class ModelInputError(Exception):
    pass


class ModelInput(namedtuple):
    def __init__(self):
        super().__init__(
            typename='ModelInput',
            field_names=('input_ids', 'token_type_ids', 'lm_labels', 'past'),
            defaults=(None, None, None, None)
        )

    def cuda(self, gpu_id):
        for name, value in self._asdict().iteritems():
            if value is not None:
                self[name] = value.cuda(gpu_id)