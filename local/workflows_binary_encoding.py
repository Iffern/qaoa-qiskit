from collections import OrderedDict

import numpy as np
from qiskit.algorithms.eigensolvers import NumPyEigensolver

M = [2, 6, 3]
K = [1, 4, 2]
T = [12, 6, 24]
d = 19


def get_time_matrix(M, T):
    r = []
    for i in M:
        tmp = []
        for j in T:
            tmp.append(j / i)
        r.append(tmp)
    return np.array(r)


def get_cost_matrix(time_matrix, K):
    m = []
    for i in range(len(time_matrix)):
        tmp = []
        for j in time_matrix[i]:
            tmp.append(K[i] * j)
        m.append(tmp)
    return m


time_matrix = np.array(get_time_matrix(M, T))
cost_matrix = np.array(get_cost_matrix(time_matrix, K))


def sample_most_likely(state_vector):
    if isinstance(state_vector, (OrderedDict, dict)):
        # get the binary string with the largest count
        binary_string = sorted(state_vector.items(), key=lambda kv: kv[1])
        probability = binary_string[-1][1]
        binary_string = binary_string[-1][0]
        return binary_string[::-1], probability
    return [], 0


optimal_key = "0000001000"


def get_stats_for_result(dict_res):
    optimal = 0
    correct = 0
    incorrect = 0
    correct_config = 0
    incorrect_config = 0
    feasible = 0
    feasible_config = 0

    if optimal_key in dict_res:
        optimal = dict_res[optimal_key]
    for key, val in dict_res.items():
        key = key[::-1]
        if is_correct(key) and key != optimal_key:
            correct += val
            correct_config += 1
        elif key != optimal_key:
            incorrect += val
            incorrect_config += 1
        if is_feasible(key):
            feasible += val
            feasible_config += 1

    return sample_most_likely(dict_res), optimal, correct, feasible, incorrect, correct_config, incorrect_config, feasible_config


def is_correct(key):
    return solution_vector_correct(key) and execution_time(key) == d


def is_feasible(key):
    return solution_vector_correct(key)


correct_machines = ['00', '01', '10']
machine_to_index = {'00': 0, '01': 1, '10': 2}


def solution_vector_correct(vector):
    task1_machine = vector[0:2]
    task2_machine = vector[2:4]
    task3_machine = vector[4:6]

    return task1_machine in correct_machines \
           and task2_machine in correct_machines \
           and task3_machine in correct_machines


def execution_time(k):
    task1_machine = machine_to_index.get(k[0:2])
    task2_machine = machine_to_index.get(k[2:4])
    task3_machine = machine_to_index.get(k[4:6])

    task1_time = time_matrix[task1_machine, 0] if task1_machine is not None else 0
    task2_time = time_matrix[task2_machine, 1] if task2_machine is not None else 0
    task3_time = time_matrix[task3_machine, 2] if task3_machine is not None else 0

    slack_sum = int(k[6]) * 8 + int(k[7]) * 4 + int(k[8]) * 2 + int(k[9]) * 1

    return task1_time + task2_time + task3_time + slack_sum


def execution_cost(k):
    task1_machine = machine_to_index.get(k[0:2])
    task2_machine = machine_to_index.get(k[2:4])
    task3_machine = machine_to_index.get(k[4:6])

    task1_cost = cost_matrix[task1_machine, 0] if task1_machine is not None else 0
    task2_cost = cost_matrix[task2_machine, 1] if task2_machine is not None else 0
    task3_cost = cost_matrix[task3_machine, 2] if task3_machine is not None else 0

    return task1_cost + task2_cost + task3_cost


def incorrect_machine_count(k):
    task1_machine = machine_to_index.get(k[0:2])
    task2_machine = machine_to_index.get(k[2:4])
    task3_machine = machine_to_index.get(k[4:6])

    return (0 if k[0:2] in correct_machines else 1) \
           + (0 if k[2:4] in correct_machines else 1) \
           + (0 if k[4:6] in correct_machines else 1)


def get_cost_model(x):
    return sum([
        cost_matrix[0, i] * (1 - x[2 * i]) * (1 - x[2 * i + 1])
        + cost_matrix[1, i] * (1 - x[2 * i]) * x[2 * i + 1]
        + cost_matrix[2, i] * x[2 * i] * (1 - x[2 * i + 1])
        for i in range(0, 3)
    ])


def get_machine_usage_model(x):
    return sum([
        x[2 * i] * x[2 * i + 1] for i in range(0, 3)
    ])


def get_deadline_model(x):
    time_sum = sum([
        time_matrix[0, i] * (1 - x[2 * i]) * (1 - x[2 * i + 1])
        + time_matrix[1, i] * (1 - x[2 * i]) * x[2 * i + 1]
        + time_matrix[2, i] * x[2 * i] * (1 - x[2 * i + 1])
        for i in range(0, 3)
    ])
    slack_sum = 8 * x[6] + 4 * x[7] + 2 * x[8] + x[9]
    time_constraint = (d - time_sum - slack_sum) ** 2

    return time_constraint


def compute_eigenvalues(qubit_op):
    count = 2 ** qubit_op.num_qubits
    eigensolver = NumPyEigensolver(k=count)
    eigensolver_result = eigensolver.compute_eigenvalues(qubit_op)
    print('state\t\ttime\tcost\tmachine use\tcorrect\teigenvalue')
    for eigenstate, eigenvalue in zip(eigensolver_result.eigenstates, eigensolver_result.eigenvalues):
        eigenstate, = eigenstate.to_dict().keys()
        eigenstate = eigenstate[::-1]
        eigenvalue = eigenvalue
        print(f'{eigenstate}\t{execution_time(eigenstate)}\t{execution_cost(eigenstate)}', end='')
        print(f'\t{incorrect_machine_count(eigenstate)}\t\t{is_correct(eigenstate)}\t{eigenvalue}')
