import pytest

from .utils_for_tests import almost_equal

from pastasolver.pasta_solver import Pasta


@pytest.mark.parametrize("filename,query,evidence,test_name,expected_lp,expected_up,normalize",[
    ("../examples/inference/alarm.lp", "calls(mary)", "", "alarm_calls_mary", 0.196, 0.196, False),
    #("../examples/inference/barber.lp", "person(j)", "", "barber_person_j", 1.0, 1.0, False),
    ("../examples/inference/bird_2_2.lp","fly_1", "", "bird_2_2_fly_1", 0.6, 0.7, False),
    ("../examples/inference/bird_4.lp","fly(1)", "", "bird_4_fly_1", 0.25, 0.5, False),
    ("../examples/inference/bird_4_different.lp", "fly(1)", "", "bird_4_different_fly_1", 0.102, 0.11, False),
    ("../examples/inference/bird_10.lp","fly(1)", "", "bird_10_fly_1", 0.127, 0.5, False),
    #("../examples/inference/burglary.lp", "qr", "", "burglary_qr", 0.5, 0.5, False),
    ("../examples/inference/certain_fact_2.lp", "a(1)", "", "certain_fact_a_1", 1.0, 1.0, False ),
    ("../examples/inference/clique.lp", "in(1)", "", "clique_in_1", 0.467, 0.933, True),
    ("../examples/inference/credal_facts.lp", "sleep(bill)", "", "credal_facts_sleep_bill", 0.0, 0.5, False),
    ("../examples/inference/disjunction.lp", "f", "", "disjunction", 0.6, 0.8, False),
    ("../examples/inference/evidence_certain.lp", "a", "", "evidence_certain_a", 0.0, 0.75, False), 
    ("../examples/inference/graph_coloring.lp", "qr", "", "graph_coloring_qr", 0.03, 1.0, False),
    ("../examples/inference/monty_hall.lp", "win_switch", "", "monty_hall", 0.667, 0.667, False), 
    ("../examples/inference/multiple_ad.lp", "qr", "", "multiple_ad", 0.098, 0.578, False),
    ("../examples/inference/path.lp","path(1,4)", "", "path_path_1_4", 0.267, 0.267, False),
    ("../examples/inference/shop.lp", "qr", "", "shop_qr", 0.096, 0.3, False),
    ("../examples/inference/sick.lp", "sick", "", "sick_sick", 0.2, 0.238, False),
    ("../examples/inference/smoke.lp", "qry", "", "smoke_qry", 0, 0.09, False),
    #("../examples/inference/smoke_sm.lp", "smokes(1)", "", "smoke_sm_smokes_1",  ),
    #("../examples/inference/smoke_2.lp", "qr", "", "smoke_qry", ),
    ("../examples/inference/smoke_3.lp", "qry", "", "smoke_3_qry", 0.3, 0.3, False),
    ("../examples/inference/transmission.lp" , "transmit(a,e)", "", "transmission_transmit_a_e", 0.772, 0.855, False),
    ("../examples/inference/viral_marketing_5.lp", "buy(5)", "", "viral_marketing_5_buy_5", 0.273, 0.29, False)
])
def test_exact_inference( 
    filename : str,
    query : str,
    evidence : str,
    test_name : str,
    expected_lp : float,
    expected_up : float,
    normalize : bool
    ):

    pasta_solver = Pasta(filename, query, evidence, normalize_prob = normalize)
    lp, up = pasta_solver.inference()

    assert almost_equal(lp,expected_lp), f"{test_name}: wrong lower probability - E: {expected_lp}, F: {lp}"
    assert almost_equal(up,expected_up), f"{test_name}: wrong upper probability - E: {expected_up}, F: {up}"

def test_barber_person_j_exit():
    with pytest.raises(SystemExit):
        test_exact_inference("../examples/inference/barber.lp", "person(j)", "", "barber_person_j", 1.0, 1.0, False)

def test_burglary_qr_exit():
    with pytest.raises(SystemExit):
        test_exact_inference("../examples/inference/burglary.lp", "qr", "", "burglary_qr", 0.5, 0.5, False)

def test_certain_fact_a1_exit():
    with pytest.raises(SystemExit):
        test_exact_inference("../examples/inference/certain_fact.lp", "a1", "", "certain_fact_a1", 1.0, 1.0, False)

def test_clique_in_1_exit():
    with pytest.raises(SystemExit):
        test_exact_inference("../examples/inference/clique.lp", "in(1)", "", "clique_in_1", 0.4666666666666667, 0.9333333333333333, False)

def test_smoke_2_qr_exit():
    with pytest.raises(SystemExit):
        test_exact_inference("../examples/inference/smoke_2.lp", "qr", "", "smoke_2_qr", 0.055408970976253295, 0.13398746701846967, False)