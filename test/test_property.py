import decimal
from hypothesis import given
import hypothesis.strategies as st
from pastasolver.pasta_solver import Pasta

QUERY = "qr"

# Strategy: decimal between 0.01 and 0.99 with 2 decimal places
decimal_strategy = st.decimals(
    min_value=decimal.Decimal("0.01"),
    max_value=decimal.Decimal("0.99"),
    places=2,
)

# Convert decimals to floats
float_strategy = decimal_strategy.map(float)

def create_pasta_program(pa: float, pb: float) -> str:
    """
    Returns a PASTA program as a string using the given pa and pb values.
    """
    return f"{pa}::a.\n{pb}::b.\nqr :- a.\nqr :- b.\n"

@given(pa=float_strategy, pb=float_strategy)
def test_property_valid_prob(pa: float, pb: float):
    """
    Check that the computed lower and upper probabilities are within [0, 1].
    """
    program = create_pasta_program(pa, pb)
    solver = Pasta("", QUERY)
    lp, up = solver.inference(program)
    assert 0 <= lp <= 1, f"Lower probability {lp} is out of bounds"
    assert 0 <= up <= 1, f"Upper probability {up} is out of bounds"

@given(pa=float_strategy, pb=float_strategy)
def test_property_same_prob(pa: float, pb: float):
    """
    Check that lower and upper probabilities are equal.
    """
    program = create_pasta_program(pa, pb)
    solver = Pasta("", QUERY)
    lp, up = solver.inference(program)
    assert lp == up, f"Lower ({lp}) and upper ({up}) probabilities differ"

@given(pa=float_strategy, pb=float_strategy)
def test_property_less_sum_prob(pa: float, pb: float):
    """
    Check that the computed probability is less than the sum pa + pb.
    """
    program = create_pasta_program(pa, pb)
    solver = Pasta("", QUERY)
    lp, _ = solver.inference(program)
    assert lp < (pa + pb), f"Lower probability {lp} is not less than pa + pb ({pa + pb})"
