from pyomo.environ import *
from pyomo.repn import generate_standard_repn
import time

# @runtime
M = ConcreteModel()
M.A = RangeSet(100000)
M.p = Param(M.A, mutable=True, initialize=1)
M.x = Var(M.A)

start = time.time()
e = sum( (M.x[i] - 1)**M.p[i] for i in M.A)
print("sum:      %f" % (time.time() - start))

start = time.time()
generate_standard_repn(e)
print("repn:     %f" % (time.time() - start))

start = time.time()
e = quicksum( (M.x[i] - 1)**M.p[i] for i in M.A)
print("quicksum: %f" % (time.time() - start))

start = time.time()
generate_standard_repn(e)
print("repn:     %f" % (time.time() - start))

# @runtime
