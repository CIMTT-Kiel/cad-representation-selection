#standard imports
from pathlib import Path

# custom imports
from invariant import InvariantCalculator
from mesher import StepMesher

def test_invariant_calculator():
    # test step file
    step_file = r"../../../data/3_primary/fabwave/Bearings/00ed2536-3d80-4f07-8851-4f49f1606498.step"

    # test invariant calculator

    # calculate invariants
    invariants = InvariantCalculator.calculate_invariants_from_step(Path(step_file))
    
    assert invariants is not None

    # test invariants
    assert invariants.mues is not None
    assert invariants.pis is not None

    # test mues
    assert len(invariants.mues) == InvariantCalculator.moment_permutations.__len__()

    # test pies
    assert len(invariants.pis) == InvariantCalculator.moment_permutations.__len__()

    # test second moments
    assert invariants.pis['pi_200'] == 1.0
    assert invariants.pis['pi_020'] == 1.0
    assert invariants.pis['pi_002'] == 1.0






test_invariant_calculator()