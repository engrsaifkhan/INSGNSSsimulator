import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'python')))

import pocket_sdr


def test_parse_rcv_pvt_solution_handles_fix():
    sol = pocket_sdr.parse_rcv_pvt_solution('2026/08/03 12:34:56.000 1.234 2.345 3.456 5/8 FIX')
    assert sol is not None
    time, pos, nsat, status = sol
    assert status == 'FIX'
    assert abs(pos[0] - 1.234 * pocket_sdr.D2R) < 1e-12
    assert abs(pos[1] - 2.345 * pocket_sdr.D2R) < 1e-12
    assert abs(pos[2] - 3.456) < 1e-12
    assert nsat == [5, 8]
