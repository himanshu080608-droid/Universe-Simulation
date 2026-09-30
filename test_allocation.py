import numpy as np

def test_allocation(N_total, fractions):
    fractions = np.array(fractions)
    fractions /= fractions.sum()
    
    N_assigned = []
    remainder = N_total
    for i in range(len(fractions)-1):
        n = int(np.round(fractions[i] * N_total))
        N_assigned.append(n)
        remainder -= n
    N_assigned.append(max(0, remainder))
    
    if sum(N_assigned) != N_total:
        print(f"Failed! N={N_total}, fractions={fractions}, assigned={N_assigned}, sum={sum(N_assigned)}")
    else:
        print(f"Passed: N={N_total}, assigned={N_assigned}")

test_allocation(3, [0.5, 0.5, 0.0])
test_allocation(10, [0.333, 0.333, 0.334])
test_allocation(5, [0.5, 0.5, 0.0])
test_allocation(7, [0.5, 0.5, 0.0])
