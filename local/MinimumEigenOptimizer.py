# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Modified version of MinimumEigenOptimizer
# by Anna Zajac, August 2023

"""A wrapper for minimum eigen solvers to be used within the optimization module."""
from typing import List, Optional, Union, cast

import numpy as np
from qiskit.algorithms.minimum_eigen_solvers import MinimumEigensolver as LegacyMinimumEigensolver
from qiskit.algorithms.minimum_eigen_solvers import (
    MinimumEigensolverResult as LegacyMinimumEigensolverResult,
)
from qiskit.algorithms.minimum_eigensolvers import (
    NumPyMinimumEigensolver,
    NumPyMinimumEigensolverResult,
    SamplingMinimumEigensolver,
    SamplingMinimumEigensolverResult,
    VQE,
)
from qiskit.opflow import OperatorBase, PauliOp, PauliSumOp
from qiskit_optimization import QiskitOptimizationError, QuadraticProgram
from qiskit_optimization.algorithms import OptimizationResult, OptimizationResultStatus, SolutionSample, \
    OptimizationAlgorithm
from qiskit_optimization.converters import QuadraticProgramConverter, QuadraticProgramToQubo
from qiskit_optimization.problems import Variable
from qiskit_optimization.translators import from_ising

MinimumEigensolver = Union[
    SamplingMinimumEigensolver, NumPyMinimumEigensolver, LegacyMinimumEigensolver
]
MinimumEigensolverResult = Union[
    SamplingMinimumEigensolverResult, NumPyMinimumEigensolverResult, LegacyMinimumEigensolverResult
]


class MinimumEigenOptimizationResult(OptimizationResult):
    """Minimum Eigen Optimizer Result."""

    def __init__(
            self,
            x: Optional[Union[List[float], np.ndarray]],
            fval: Optional[float],
            variables: List[Variable],
            status: OptimizationResultStatus,
            samples: Optional[List[SolutionSample]] = None,
            min_eigen_solver_result: Optional[MinimumEigensolverResult] = None,
            raw_samples: Optional[List[SolutionSample]] = None,
    ) -> None:
        """
        Args:
            x: the optimal value found by ``MinimumEigensolver``.
            fval: the optimal function value.
            variables: the list of variables of the optimization problem.
            status: the termination status of the optimization algorithm.
            min_eigen_solver_result: the result obtained from the underlying algorithm.
            samples: the x values, the objective function value of the original problem,
                the probability, and the status of sampling.
            raw_samples: the x values of the QUBO, the objective function value of the QUBO,
                and the probability of sampling.
        """
        super().__init__(
            x=x,
            fval=fval,
            variables=variables,
            status=status,
            raw_results=None,
            samples=samples,
        )
        self._min_eigen_solver_result = min_eigen_solver_result
        self._raw_samples = raw_samples

    @property
    def min_eigen_solver_result(self) -> MinimumEigensolverResult:
        """Returns a result object obtained from the instance of :class:`MinimumEigensolver`."""
        return self._min_eigen_solver_result

    @property
    def raw_samples(self) -> Optional[List[SolutionSample]]:
        """Returns the list of raw solution samples of ``MinimumEigensolver``.

        Returns:
            The list of raw solution samples of ``MinimumEigensolver``.
        """
        return self._raw_samples


class MinimumEigenOptimizer(OptimizationAlgorithm):
    """A wrapper for minimum eigen solvers.

    This class provides a wrapper for minimum eigen solvers from Qiskit to be used within
    the optimization module.
    It assumes a problem consisting only of binary or integer variables as well as linear equality
    constraints thereof. It converts such a problem into a Quadratic Unconstrained Binary
    Optimization (QUBO) problem by expanding integer variables into binary variables and by adding
    the linear equality constraints as weighted penalty terms to the objective function. The
    resulting QUBO is then translated into an Ising Hamiltonian whose minimal eigen vector and
    corresponding eigenstate correspond to the optimal solution of the original optimization
    problem. The provided minimum eigen solver is then used to approximate the ground state of the
    Hamiltonian to find a good solution for the optimization problem.

    Examples:
        Outline of how to use this class:

    .. code-block::

        from qiskit.algorithms.minimum_eigensolver import QAOA
        from qiskit_optimization.problems import QuadraticProgram
        from qiskit_optimization.algorithms import MinimumEigenOptimizer
        problem = QuadraticProgram()
        # specify problem here
        # specify minimum eigen solver to be used, e.g., QAOA
        qaoa = QAOA(...)
        optimizer = MinimumEigenOptimizer(qaoa)
        result = optimizer.solve(problem)
    """

    def __init__(
            self,
            min_eigen_solver: MinimumEigensolver,
            penalty: Optional[float] = None,
            converters: Optional[
                Union[QuadraticProgramConverter, List[QuadraticProgramConverter]]
            ] = None,
    ) -> None:
        """
        This initializer takes the minimum eigen solver to be used to approximate the ground state
        of the resulting Hamiltonian as well as a optional penalty factor to scale penalty terms
        representing linear equality constraints. If no penalty factor is provided, a default
        is computed during the algorithm.

        Args:
            min_eigen_solver: The eigen solver to find the ground state of the Hamiltonian.
            penalty: The penalty factor to be used, or ``None`` for applying a default logic.
            converters: The converters to use for converting a problem into a different form.
                By default, when None is specified, an internally created instance of
                :class:`~qiskit_optimization.converters.QuadraticProgramToQubo` will be used.

        Raises:
            TypeError: If minimum eigensolver has an invalid type.
            TypeError: When one of converters has an invalid type.
            QiskitOptimizationError: When the minimum eigensolver does not return an eigenstate.
        """
        self._min_eigen_solver = min_eigen_solver
        self._penalty = penalty

        self._converters = self._prepare_converters(converters, penalty)

    def get_compatibility_msg(self, problem: QuadraticProgram) -> str:
        """Checks whether a given problem can be solved with this optimizer.

        Checks whether the given problem is compatible, i.e., whether the problem can be converted
        to a QUBO, and otherwise, returns a message explaining the incompatibility.

        Args:
            problem: The optimization problem to check compatibility.

        Returns:
            A message describing the incompatibility.
        """
        return QuadraticProgramToQubo.get_compatibility_msg(problem)

    @property
    def min_eigen_solver(self) -> MinimumEigensolver:
        """Returns the minimum eigensolver."""
        return self._min_eigen_solver

    @min_eigen_solver.setter
    def min_eigen_solver(self, min_eigen_solver: MinimumEigensolver) -> None:
        """Sets the minimum eigensolver."""
        self._min_eigen_solver = min_eigen_solver

    def solve(self, problem: QuadraticProgram) -> MinimumEigenOptimizationResult:
        """Tries to solves the given problem using the optimizer.

        Runs the optimizer to try to solve the optimization problem.

        Args:
            problem: The problem to be solved.

        Returns:
            The result of the optimizer applied to the problem.

        Raises:
            QiskitOptimizationError: If problem not compatible.
        """
        self._verify_compatibility(problem)

        # convert problem to QUBO minimization problem
        problem_ = self._convert(problem, self._converters)

        # construct operator and offset
        operator, offset = problem_.to_ising()

        return self._solve_internal(operator, offset, problem_, problem)

    def solve_hamiltonian(self, operator: OperatorBase, offset: float,
                          problem: QuadraticProgram = None) -> MinimumEigenOptimizationResult:
        """Tries to solves the given problem using the optimizer.

        Runs the optimizer to try to solve the optimization problem.

        Args:
            problem: The problem to be solved.

        Returns:
            The result of the optimizer applied to the problem.

        Raises:
            QiskitOptimizationError: If problem not compatible.
        """
        if problem is None:
            problem = from_ising(operator, offset)

        self._verify_compatibility(problem)

        # convert problem to QUBO minimization problem
        problem_ = self._convert(problem, self._converters)

        return self._solve_internal(operator, offset, problem_, problem)

    def _solve_internal(
            self,
            operator: OperatorBase,
            offset: float,
            converted_problem: QuadraticProgram,
            original_problem: QuadraticProgram,
    ):
        # only try to solve non-empty Ising Hamiltonians
        eigen_result: Optional[MinimumEigensolverResult] = None
        if operator.num_qubits > 0:
            # NumPyEigensolver does not accept PauliOp but PauliSumOp
            if isinstance(operator, PauliOp):
                operator = PauliSumOp.from_list([(operator.primitive.to_label(), operator.coeff)])
            # approximate ground state of operator using min eigen solver
            eigen_result = self._min_eigen_solver.compute_minimum_eigenvalue(operator)
            # analyze results
            if eigen_result.eigenstate is not None and isinstance(eigen_result.eigenstate, list):
                n = len(eigen_result.eigenstate)
                raw_samples = [SolutionSample(x=np.fromstring(e, 'u1') - ord('0'), fval=eigen_result.eigenvalue[idx],
                                              probability=(1 / n),
                                              status=OptimizationResultStatus.SUCCESS)
                               for idx, e in enumerate(eigen_result.eigenstate)]
            elif eigen_result.eigenstate is not None:
                raw_samples = self._eigenvector_to_solutions(
                    eigen_result.eigenstate, converted_problem
                )
                raw_samples.sort(key=lambda x: x.fval)
        else:
            # if Hamiltonian is empty, then the objective function is constant to the offset
            x = np.zeros(converted_problem.get_num_binary_vars())
            fval = offset
            raw_samples = [SolutionSample(x, fval, 1.0, OptimizationResultStatus.SUCCESS)]

        if raw_samples is None:
            # if not function value is given, then something went wrong, e.g., a
            # NumPyMinimumEigensolver has been configured with an infeasible filter criterion.
            return MinimumEigenOptimizationResult(
                x=None,
                fval=None,
                variables=original_problem.variables,
                status=OptimizationResultStatus.FAILURE,
                samples=None,
                raw_samples=None,
                min_eigen_solver_result=eigen_result,
            )

        # translate result back to integers and eventually maximization
        samples, best_raw = self._interpret_samples(original_problem, raw_samples, self._converters)
        return cast(
            MinimumEigenOptimizationResult,
            self._interpret(
                x=best_raw.x,
                converters=self._converters,
                problem=original_problem,
                result_class=MinimumEigenOptimizationResult,
                samples=samples,
                raw_samples=raw_samples,
                min_eigen_solver_result=eigen_result,
            ),
        )
