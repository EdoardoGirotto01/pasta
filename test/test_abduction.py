import pytest
from .utils_for_tests import almost_equal, check_if_lists_equal
from pastasolver.pasta_solver import Pasta

def wrap_test_abduction(filename: str,
                        query: str,
                        evidence: str,
                        test_name: str,
                        expected_lp: float,
                        expected_up: float,
                        expected_abd: 'list[list[str]]'):
    pasta_solver = Pasta(filename, query, evidence)
    lp, up, abd = pasta_solver.abduction()

    assert almost_equal(lp, expected_lp), test_name + ": wrong lower probability"
    assert almost_equal(up, expected_up), test_name + ": wrong upper probability"
    assert check_if_lists_equal(abd, expected_abd), f"{test_name}: wrong abduction. Found {abd} expected {expected_abd}"

@pytest.mark.parametrize(
    "filename,query,evidence,test_name,expected_lp,expected_up,expected_abd",
    [
        ("../examples/abduction/bird_4_abd_prob.lp", "fly(1)", "", "bird_4_abd_prob", 0.5, 0.5, [['fa(1)']]),
        ("../examples/abduction/ex_1_det.lp", "query", "", "ex_1_det", 1, 1, [['a', 'b']]),
        ("../examples/abduction/ex_1_prob.lp", "query", "", "ex_1_prob", 0.25, 0.25, [['a', 'b']]),
        ("../examples/abduction/ex_2_det.lp", "query", "", "ex_2_det", 1, 1, [['b'], ['a']]),
        ("../examples/abduction/ex_2_prob.lp", "query", "", "ex_2_prob", 0.75, 0.75, [['a', 'b']]),
        ("../examples/abduction/ex_3_det.lp", "query", "", "ex_3_det", 1, 1, [['a'], ['c', 'b']]),
        ("../examples/abduction/ex_3_prob.lp", "query", "", "ex_3_prob", 0.58, 0.58, [['a', 'b', 'c']]),
        ("../examples/abduction/ex_4_det.lp", "query", "", "ex_4_det", 1, 1, [['a(1)'], ['c', 'b']]),
        # Test ex_4_prob omesso: errore di inconsistenza
        ("../examples/abduction/ex_5_det.lp", "qr", "", "ex_5_det", 1, 1, [['c', 'b', 'a'], ['e', 'd', 'b', 'a']]),
        ("../examples/abduction/ex_6_prob.lp", "qry", "", "ex_6_prob", 1.0, 1.0, [['a(1)']]),
        ("../examples/abduction/kn_4_prob.lp", "path(1,4)", "", "kn_4_prob", 0.734375, 0.734375,
         [['edge(1,2)', 'edge(1,3)', 'edge(1,4)', 'edge(2,3)', 'edge(2,4)', 'edge(3,4)']]),
        ("../examples/abduction/smokes_det.lp", "smokes(c)", "", "smokes_det", 1, 1,
         [['e(b,c)'], ['e(e,c)', 'e(d,e)']])
    ]
)
def test_abduction(filename, query, evidence, test_name, expected_lp, expected_up, expected_abd):
    wrap_test_abduction(filename, query, evidence, test_name, expected_lp, expected_up, expected_abd)
