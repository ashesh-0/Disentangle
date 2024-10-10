import torch


def enable_epistemic_uncertainty_computation_mode(model, inp = None):
    """
    After this is called, the uncertainty in prediction will essentially be epistemic uncertainty.
    This is because the dropout layers will be enabled during test-time and so we do Monte Carlo Dropout.
    We disable the sampling from the posterior distribution in order to eliminate the aleatoric uncertainty.
    We also disable non-deterministic behaviour in Pytorch to ensure that it does not contribute to the uncertainty.
    """
    torch.use_deterministic_algorithms(True)
    enable_dropout(model)
    model.non_stochastic_version = True
    print('')
    print('')
    print("Epistemic uncertainty computation mode enabled.")
    print('LVAE stochasticity disabled')
    print('Pytorch deterministic algorithms enabled')
    print('Dropout layers enabled')

    if inp is not None:
        epistemic_uncertainty_sanity_check(model, inp)
        print('Sanity check: Passed')
    
    print('')
    print('')


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def disable_dropout(model):
    """ Function to disable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.eval()


def epistemic_uncertainty_sanity_check(model, inp:torch.Tensor):
    disable_dropout(model)
    with torch.no_grad():
        out1,_ = model(inp)
        out2,_ = model(inp)
        assert torch.allclose(out1, out2), "There should be uncertainty only from dropout. This is not the case now."

    enable_dropout(model)
    with torch.no_grad():
        out1,_ = model(inp)
        out2,_ = model(inp)
        assert not torch.allclose(out1, out2), "Now, there should be uncertainty"
