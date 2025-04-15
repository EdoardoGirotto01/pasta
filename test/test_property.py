import sys
import os
import tempfile
import decimal
from hypothesis import given
import hypothesis.strategies as st
from pastasolver.pasta_solver import Pasta

# Ensure that "pastasolver" can be imported (if needed)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Default settings for the Pasta solver
QUERY = "qr"
EVIDENCE = None
NORMALIZE = True

def create_pasta_content(pa: float, pb: float) -> str:
    """
    Returns the string content of the ASP program to be used by Pasta.
    For example:
      0.25::a.
      0.45::b.
      qr:- a.
      qr:- b.
    """
    return f"{pa}::a.\n{pb}::b.\nqr:- a.\nqr:- b.\n"

# 1) Define a strategy producing decimals in [0.01, 0.99] with 2 decimal places
decimal_strategy = st.decimals(
    min_value=decimal.Decimal("0.01"),
    max_value=decimal.Decimal("0.99"),
    places=2,
)

# 2) Convert those decimals to floats with map(...)
float_strategy = decimal_strategy.map(float)

@given(
    pa=float_strategy,
    pb=float_strategy
)
def test_property_valid_prob(pa: float, pb: float):
    """
    Checks that the probabilities (lp and up) returned by Pasta are in [0,1].
    We avoid pa or pb being 0.0 or 1.0 or any extreme scientific notation.
    """
    content = create_pasta_content(pa, pb)
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".lp", delete=False) as tmp:
        # Write our generated ASP content into the temp file
        tmp.write(content)
        tmp.flush()
        tmp_name = tmp.name

    try:
        # Pass the file path to the Pasta solver
        pasta_solver = Pasta(tmp_name, QUERY, EVIDENCE, normalize_prob=NORMALIZE)
        lp, up = pasta_solver.inference()

        # Assert that lp, up are between 0 and 1
        assert 0 <= lp <= 1, f"lp = {lp} is not in [0,1]"
        assert 0 <= up <= 1, f"up = {up} is not in [0,1]"
    finally:
        # Remove the temp file
        os.remove(tmp_name)

@given(
    pa=float_strategy,
    pb=float_strategy
)
def test_property_same_prob(pa: float, pb: float):
    """
    Checks that lp == up (if the system produces a single point probability).
    """
    content = create_pasta_content(pa, pb)
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".lp", delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_name = tmp.name

    try:
        pasta_solver = Pasta(tmp_name, QUERY, EVIDENCE, normalize_prob=NORMALIZE)
        lp, up = pasta_solver.inference()
        # Ensure the lower and upper probabilities match
        assert lp == up, f"lp ({lp}) != up ({up})"
    finally:
        os.remove(tmp_name)

@given(
    pa=float_strategy,
    pb=float_strategy
)
def test_property_less_sum_prob(pa: float, pb: float):
    """
    Verifies that lp < (pa + pb), assuming lp == up.
    """
    content = create_pasta_content(pa, pb)
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".lp", delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_name = tmp.name

    try:
        pasta_solver = Pasta(tmp_name, QUERY, EVIDENCE, normalize_prob=NORMALIZE)
        lp, up = pasta_solver.inference()
        # Check that lp is strictly less than the sum of pa and pb
        assert lp < (pa + pb), f"lp ({lp}) >= pa+pb ({pa+pb})"
    finally:
        os.remove(tmp_name)
# DEBUG: questa riga serve a forzare una modifica

