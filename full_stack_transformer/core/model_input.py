from collections import namedtuple


class ModelInputError(Exception):
    pass


ModelInput = namedtuple(
    typename='ModelInput',
    field_names=('input_ids', 'token_type_ids', 'lm_labels', 'past'),
    defaults=(None, None, None, None)
)
