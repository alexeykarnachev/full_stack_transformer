from collections import namedtuple


class ModelInputError(Exception):
    pass


ModelInput = namedtuple(
    typename='ModelInput',
    field_names=('input_ids', 'token_type_ids', 'lm_labels', 'past'),
    defaults=(None, None, None, None)
)


def _cuda(self, gpu_id):
    for name, value in self._asdict().items():
        if value is not None:
            self[name] = value.cuda(gpu_id)


ModelInput.cuda = _cuda
