import functools

import sympy
from qiskit.opflow import SummedOp, Z, I


class PolynomialProgram:

    def __init__(self, num_variables):
        self.x = sympy.symbols(' '.join([f'x{i}' for i in range(num_variables)]))
        self.objective = 0

    def add_objective(self, objective: sympy.core.expr.Expr, weight=1):
        self.objective += weight * sympy.simplify(sympy.expand(objective))

    @staticmethod
    def _create_z(z_indices, n):
        return functools.reduce(lambda a, b: a ^ b, reversed([Z if i in z_indices else I for i in range(n)]))

    def _get_summed_ops(self, coefficient_dict):
        n = len(self.x)
        ops = []
        offset = 0

        for summand, coefficient in coefficient_dict.items():
            coefficient = float(coefficient)
            if summand.is_Number:
                offset += float(summand)
            elif summand.is_Symbol:
                index = int(summand.name[1:])
                ops += [coefficient * self._create_z([index], n)]
            elif summand.is_Mul:
                indices = [
                    int(factor.base.name[1:])
                    if factor.is_Pow
                    else int(factor.name[1:])
                    for factor
                    in summand.args
                    if not factor.is_Pow or factor.exp % 2 == 1
                ]
                ops += [coefficient * self._create_z(indices, n)]
            elif summand.is_Pow:
                indices = []
                if summand.exp % 2 == 1:
                    indices += [int(summand.base.name[1:])]
                ops += [coefficient * self._create_z(indices, n)]
            else:
                raise RuntimeError(f'Error: unknown summand: {summand}')

        return SummedOp(ops).reduce(), offset

    def to_ising(self):
        simplified_objective = sympy.simplify(self.objective)
        z = sympy.symbols(' '.join([f'z{i}' for i in range(len(self.x))]))

        for x_i, z_i in zip(self.x, z):
            simplified_objective = simplified_objective.replace(x_i, .5 * (1 - x_i))

        coefficients = sympy.simplify(simplified_objective).as_coefficients_dict()
        hamiltonian, offset = self._get_summed_ops(coefficients)

        return hamiltonian, offset