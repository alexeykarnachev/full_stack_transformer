from collections import namedtuple


class ModelInputError(Exception):
    pass


ModelInput = namedtuple(
    typename='ModelInput',
    field_names=('input_ids', 'token_type_ids', 'lm_labels', 'past'),
    defaults=(None, None, None, None)
)


def _cuda(self, gpu_id):
    d = self._asdict()
    for name, value in d.items():
        if value is not None:
            d[name] = value.cuda(gpu_id)

    return self


ModelInput.cuda = _cuda
