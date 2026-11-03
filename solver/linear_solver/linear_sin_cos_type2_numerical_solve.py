import numpy as np
from typing import Tuple, Optional, List
from python_run_import import try_solve_linear_type2, try_solve_linear_type2_specific_rows


def try_solve_linear_type2_combined(
        A_combined: np.ndarray,
        rows2try: List[Tuple[int, int, int]]) -> Optional[Tuple[float, float]]:
    assert len(rows2try) > 0
    for rows_i in range(len(rows2try)):
        row_0, row_1, row_2 = rows2try[rows_i]
        A_tmp_sol = try_solve_linear_type2_specific_rows(A_combined, row_0, row_1, row_2)
        if A_tmp_sol is not None:
            return A_tmp_sol
    return None
