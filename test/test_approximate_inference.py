import argparse
import pytest

from pastasolver.pasta_solver import Pasta
from .utils_for_tests import ArgumentsTest, almost_equal

# Parametrized test for approximate inference.
# For test cases that are expected to raise an exception (e.g., due to an inconsistent world),
# the ArgumentsTest object should have expected_lp and expected_up set to None and expected_exception set to the exception type.
@pytest.mark.parametrize("parameters", [
    # Test case expected to raise an exception:
    ArgumentsTest("alarm_calls_mary", "../examples/inference/alarm.lp", "calls(mary)", None, None, expected_exception=Exception),
    
    # Test cases expecting numerical results:
    ArgumentsTest("barber_person_j", "../examples/inference/barber.lp", "person(j)", 1.0, 1.0),
    ArgumentsTest("bird_2_2_fly_1", "../examples/inference/bird_2_2.lp", "fly_1", 0.6, 0.7),
    ArgumentsTest("bird_4_fly_1", "../examples/inference/bird_4.lp", "fly(1)", 0.25, 0.5),
    ArgumentsTest("bird_4_different_fly_1", "../examples/inference/bird_4_different.lp", "fly(1)", 0.102, 0.11),
    
    # Test case expected to raise an exception:
    ArgumentsTest("bird_10_fly_1", "../examples/inference/bird_10.lp", "fly(1)", None, None, expected_exception=Exception),
    
    # Remaining tests:
    ArgumentsTest("burglary_qr", "../examples/inference/burglary.lp", "qr", 0.0, 0.0),
    ArgumentsTest("certain_fact_a_1", "../examples/inference/certain_fact_2.lp", "a(1)", 1.0, 1.0),
    
    # Test case expected to raise an exception:
    ArgumentsTest("clique_in_1", "../examples/inference/clique.lp", "in(1)", None, None, expected_exception=Exception),
    
    ArgumentsTest("credal_facts_sleep_bill", "../examples/inference/credal_facts.lp", "sleep(bill)", 0.0, 0.496),
    ArgumentsTest("disjunction", "../examples/inference/disjunction.lp", "f", 0.6, 0.8),
    ArgumentsTest("evidence_certain_a", "../examples/inference/evidence_certain.lp", "a", 0.0, 0.752),
    ArgumentsTest("graph_coloring_qr", "../examples/inference/graph_coloring.lp", "qr", 0.03, 1.0),
    ArgumentsTest("monty_hall", "../examples/inference/monty_hall.lp", "win_switch", 0.667, 0.667),
    ArgumentsTest("multiple_ad", "../examples/inference/multiple_ad.lp", "qr", 0.098, 0.578),
    ArgumentsTest("path_path_1_4", "../examples/inference/path.lp", "path(1,4)", 0.267, 0.267),
    ArgumentsTest("shop_qr", "../examples/inference/shop.lp", "qr", 0.096, 0.3),
    ArgumentsTest("sick_sick", "../examples/inference/sick.lp", "sick", 0.2, 0.238),
    ArgumentsTest("smoke_qry", "../examples/inference/smoke.lp", "qry", 0, 0.09),
    ArgumentsTest("smoke_3_qry", "../examples/inference/smoke_3.lp", "qry", 0.3, 0.3),
    ArgumentsTest("transmission_transmit_a_e", "../examples/inference/transmission.lp", "transmit(a,e)", 0.714, 0.794),
    ArgumentsTest("viral_marketing_5_buy_5", "../examples/inference/viral_marketing_5.lp", "buy(5)", 0.273, 0.29)
])
def test_approximate_inference(parameters: ArgumentsTest):
    # Initialize the Pasta solver with the given filename and query, using 10,000 samples.
    pasta_solver = Pasta(parameters.filename, parameters.query, samples=10_000)

    # Set up solver arguments.
    args = argparse.Namespace()
    args.rejection = parameters.rejection
    args.mh = parameters.mh
    args.gibbs = parameters.gibbs
    args.approximate_hybrid = False
    args.normalize = True

    # If an exception is expected, verify that it is raised.
    if hasattr(parameters, "expected_exception") and parameters.expected_exception is not None:
        with pytest.raises(parameters.expected_exception):
            pasta_solver.approximate_solve(args)
    else:
        # Otherwise, get the results and perform the assertions.
        lp, up = pasta_solver.approximate_solve(args)
        assert almost_equal(lp, parameters.expected_lp), f"{parameters.test_name}: wrong lower probability - E: {parameters.expected_lp}, F: {lp}"
        assert almost_equal(up, parameters.expected_up), f"{parameters.test_name}: wrong upper probability - E: {parameters.expected_up}, F: {up}"
