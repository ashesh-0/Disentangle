def get_objects_of_type(model, pytorch_type=None, except_pytorch_type=None, name_prefix=''):
    """
    Returns a list of all objects of a given type in a model. For debugging, we also return the names
    """
    both_not_none = pytorch_type is not None and except_pytorch_type is not None
    assert not both_not_none, 'pytorch_type and except_pytorch_type cannot both be not None'

    named_children = list(model.named_children())
    if len(named_children) == 0:
        if except_pytorch_type is not None and isinstance(model, except_pytorch_type):
            return [], []

        elif (pytorch_type is None) or isinstance(model, pytorch_type):
            return [model], [name_prefix]
        else:
            return [], []
    else:
        params = []
        names = []
        for name, child in named_children:
            new_name_prefix = name_prefix + '.' + name if name_prefix != '' else name
            param_child, name_child = get_objects_of_type(child, pytorch_type=pytorch_type, name_prefix=new_name_prefix)
            params += param_child
            names += name_child
        return params, names


def get_params_of_type(model, pytorch_type=None, except_pytorch_type=None, just_bias=False):
    """
    Returns a list of all parameters of a given type in a model. For debugging, we also return the names
    """
    objparams, objnames = get_objects_of_type(model, pytorch_type=pytorch_type, except_pytorch_type=except_pytorch_type)
    params = []
    for obj in objparams:
        for name, param in obj.named_parameters():
            if just_bias:
                if name.split('.')[-1] == 'bias':
                    params.append(param)
            else:
                params.append(param)
    return params


if __name__ == '__main__':
    import torch

    objparams = get_params_of_type(model, pytorch_type=None, just_bias=True)
    diff_params = []
    for name, param in model.named_parameters():
        if name.split('.')[-1] == 'bias':
            diff_params.append(param)

    assert len(objparams) == len(diff_params)
    for p1, p2 in zip(objparams, diff_params):
        assert torch.all(p1 == p2)
