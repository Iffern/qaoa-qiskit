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


optimal_key = "0000001010001"


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


correct_machines = ['100', '010', '001']
machine_to_index = {'100': 0, '010': 1, '001': 2}


def solution_vector_correct(vector):
    task1_machine = vector[0:3]
    task2_machine = vector[3:6]
    task3_machine = vector[6:9]

    return task1_machine in correct_machines \
           and task2_machine in correct_machines \
           and task3_machine in correct_machines


def execution_time(k):
    task1_machine = machine_to_index.get(k[0:3])
    task2_machine = machine_to_index.get(k[3:6])
    task3_machine = machine_to_index.get(k[6:9])

    task1_time = time_matrix[task1_machine, 0] if task1_machine is not None else 0
    task2_time = time_matrix[task2_machine, 1] if task2_machine is not None else 0
    task3_time = time_matrix[task3_machine, 2] if task3_machine is not None else 0

    slack_sum = int(k[9]) * 8 + int(k[10]) * 4 + int(k[11]) * 2 + int(k[12]) * 1

    return task1_time + task2_time + task3_time + slack_sum


def execution_cost(k):
    task1_machine = machine_to_index.get(k[0:3])
    task2_machine = machine_to_index.get(k[3:6])
    task3_machine = machine_to_index.get(k[6:9])

    task1_cost = cost_matrix[task1_machine, 0] if task1_machine is not None else 0
    task2_cost = cost_matrix[task2_machine, 1] if task2_machine is not None else 0
    task3_cost = cost_matrix[task3_machine, 2] if task3_machine is not None else 0

    return task1_cost + task2_cost + task3_cost


def incorrect_machine_count(k):
    task1_machine = machine_to_index.get(k[0:3])
    task2_machine = machine_to_index.get(k[3:6])
    task3_machine = machine_to_index.get(k[6:9])

    return (0 if k[0:3] in correct_machines else 1) \
           + (0 if k[3:6] in correct_machines else 1) \
           + (0 if k[6:9] in correct_machines else 1)


def get_cost_model(x):
    return sum(
        sum([
            cost_matrix[machine_index, task_index] * x[task_index * 3 + machine_index]
            for machine_index
            in range(3)
        ])
        for task_index
        in range(3)
    )


def get_machine_usage_model(x):
    return sum(
        (1 - sum([x[task_index * 3 + machine_index] for machine_index in range(3)])) ** 2
        for task_index
        in range(3)
    )


def get_deadline_model(x):
    time_sum = sum(
        sum([
            time_matrix[machine_index, task_index] * x[task_index * 3 + machine_index]
            for machine_index
            in range(3)
        ])
        for task_index
        in range(3)
    )
    slack_sum = 8 * x[9] + 4 * x[10] + 2 * x[11] + x[12]
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

def get_indices():
    states = []

    def generate_state(curr_string):
        if len(curr_string) == 13:
            states.append(curr_string)
            return
        elif len(curr_string) >= 9:
            generate_state(curr_string + '0')
            generate_state(curr_string + '1')
        else:
            for machine in correct_machines:
                generate_state(curr_string + machine)

    for machine in correct_machines:
        generate_state(machine)

    states_int = [int(i, 2) for i in states]
    return states_int
