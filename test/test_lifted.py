import pytest
from .utils_for_tests import almost_equal
import pastasolver.lifted.lifted as lft

@pytest.mark.parametrize(
    "params, lower_val, upper_val, expected_lp, expected_up, test_name",
    [
        # Test 1: [1, 3]
        ([1, 3], 40, None, 0.0576, 0.16, "test_cxy_ax_bxy_multiple_bi_1_lower"),
        ([1, 3], 0, 70, 0, 0.10240, "test_cxy_ax_bxy_multiple_bi_1_upper"),
        
        # Test 2: [1, 1, 2, 2]
        ([1, 1, 2, 2], 40, None, 0.071424, 0.16, "test_cxy_ax_bxy_multiple_bi_2_lower"),
        ([1, 1, 2, 2], 0, 70, 0, 0.088576, "test_cxy_ax_bxy_multiple_bi_2_upper"),
        
        # Test 3: [1, 1, 1, 2, 2, 1]
        ([1, 1, 1, 2, 2, 1], 40, None, 0.059996160, 0.16, "test_cxy_ax_bxy_multiple_bi_3_lower"),
        ([1, 1, 1, 2, 2, 1], 0, 70, 0, 0.10000384, "test_cxy_ax_bxy_multiple_bi_3_upper"),
        
        # Test 4: [1, 1, 3, 2]
        ([1, 1, 3, 2], 40, None, 0.0428544, 0.16, "test_cxy_ax_bxy_multiple_bi_4_lower"),
        ([1, 1, 3, 2], 0, 70, 0, 0.1171456, "test_cxy_ax_bxy_multiple_bi_4_upper"),
        
        # Test 5: [1, 1, 1, 3, 2, 1]
        ([1, 1, 1, 3, 2, 1], 40, None, 0.035997696, 0.16, "test_cxy_ax_bxy_multiple_bi_5_lower"),
        ([1, 1, 1, 3, 2, 1], 0, 70, 0, 0.1240023, "test_cxy_ax_bxy_multiple_bi_5_upper"),
        
        # Test 6: [1, 1, 1, 1, 1, 3, 2, 2, 1, 1]
        ([1, 1, 1, 1, 1, 3, 2, 2, 1, 1], 10, None, 0.02249712, 0.16, "test_cxy_ax_bxy_multiple_bi_6_lower"),
        ([1, 1, 1, 1, 1, 3, 2, 2, 1, 1], 0, 80, 0, 0.137502879, "test_cxy_ax_bxy_multiple_bi_6_upper"),
    ]
)
def test_cxy_ax_bxy_multiple_bi(params, lower_val, upper_val, expected_lp, expected_up, test_name):
    """
    Parameterized test for the function cxy_ax_bxy_multiple_bi.
    
    If 'upper_val' is None, the test is executed by passing only the 'lower' parameter;
    otherwise both 'lower' and 'upper' parameters are passed.
    """
    if upper_val is None:
        lp, up, _, _ = lft.cxy_ax_bxy_multiple_bi(0.4, params, lower=lower_val)
    else:
        lp, up, _, _ = lft.cxy_ax_bxy_multiple_bi(0.4, params, lower=lower_val, upper=upper_val)
    
    assert almost_equal(lp, expected_lp), (
        f"{test_name}: lower probability incorrect - expected: {expected_lp}, got: {lp}"
    )
    assert almost_equal(up, expected_up), (
        f"{test_name}: upper probability incorrect - expected: {expected_up}, got: {up}"
    )