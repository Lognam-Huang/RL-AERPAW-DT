import numpy as np
from ortools.sat.python import cp_model

def assign_gus(num_gus, num_uavs, desired_throughputs, capacities, path_qualities, alpha=0.5, beta=0.5):
    """
    Args:
        num_gus (int): Number of ground users.
        num_uavs (int): Number of UAVs.
        desired_throughputs (np.array(int)): Desired throughput for each ground user.
        capacities (np.array(int)): Total throughput capacity for each UAV.
        path_qualities (np.array(int)): Theoretical maximum throughput value between each ground user and UAV.
        alpha (float): Optimization coefficient for the throughput maximization objective.
        beta (float): Optimization coefficient for the minimum number of GUs per UAV objective.
    
    Returns:
        assignments (list(list(int))): assignments[i] contains the list of GUs assigned to UAV i.
        total_throughput (int): The total theoretical maximum throughput of all UAV-GU connections.
        coverage (float): The fraction of GUs that are assigned to a UAV.
        min_gus (int): The minimum number of GUs assigned to any UAV.
    """

    # Ensuring all inputs are integers, necessary for the solver
    try:
        desired_throughputs = desired_throughputs.astype(np.int64)
        capacities = capacities.astype(np.int64)
        path_qualities = path_qualities.astype(np.int64)
    except ValueError:
        raise ValueError("Input data must be of integer type or broadcastable.")

    model = cp_model.CpModel()
    gus = range(num_gus)
    uavs = range(num_uavs)
    
    x = []
    for i in gus:
        x.append([])
        for j in uavs:
            x[i].append(model.NewBoolVar(""))
    x = np.array(x)

    # Assigning each gu to at most one uav
    for i in gus:
        model.Add(np.sum(x[i, :]) <= 1)

    # Ensuring each uav's assigned gus do not exceed its total capacity
    for j in uavs:
        model.Add(np.dot(desired_throughputs, x[:, j]) <= capacities[j])

    # Defining the task counts for each uav
    assignment_counts = np.array([model.NewIntVar(0, num_gus, "") for j in uavs])
    for j in uavs:
        model.Add(assignment_counts[j] == np.sum(x[:, j]))

    # Defining the minimum number of gus across all uavs
    min_gus = model.NewIntVar(0, num_gus, "")
    for j in uavs:
        model.Add(assignment_counts[j] >= min_gus)

    # Maximizing the combined objective
    model.Maximize(np.sum(path_qualities * x) * alpha + min_gus * beta)

    # Solving
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    # If the model failed to solve the problem
    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        raise ValueError("Input is unsolvable or infeasible.")

    # Getting uav assignments
    assignments = []
    for j in uavs:
        assignments.append([])
        for i in gus:
            if solver.Value(x[i][j]):
                assignments[j].append(i)

    # Undoing objective function to find the total throughput
    total_throughput = (solver.ObjectiveValue() - solver.Value(min_gus) * beta) / alpha

    return assignments, total_throughput

# Example Usage
if __name__ == "__main__":
    num_gus = 1000
    num_uavs = 4
    weights = np.random.rand(num_gus) * 5  # Desired throughputs for each ground user
    capacities = np.random.rand(num_uavs) * 1250  # Total throughput capacities for each UAV
    path_qualities = np.random.rand(num_gus, num_uavs) * 20 # Path qualities between each ground user and UAV

    # Solve with weights (adjust based on priorities)
    assignments, total_throughput = assign_gus(num_gus, num_uavs, weights, capacities, path_qualities, alpha=10.0, beta=1.0)

    print("Assignments:", assignments)
    print("Total Throughput:", total_throughput)


def assignGUs(self, scores):
        """
        Args:
        scores (np.array(num_rx, num_tx)): An array of path qualities, should be theoretical maximum throughput values in bit/sec, dtype should be np.float64

        Returns:
            tuple(list(num_tx, num_rx_assigned), float, float): A tuple where the first value is a list of the indices of the ground users assigned to each UAV,
            the second value is the total path quality of assigned connections,
            the third value is the coverage percentage, the proportion of GUs serviced by the UAVs
        """

        solver = pywraplp.Solver.CreateSolver('GLOP')
        
        # Create variables
        x = []
        for i in range(self.n_rx):
            row = []
            for j in range(self.n_tx):
                row.append(solver.IntVar(0, 1, ""))
            x.append(row)
        x = np.array(x)
        capacities = np.array([x.num_channels for x in self.uavs.values()])

        # Constraints: Each user assigned to at most one UAV
        for i in range(self.n_rx):
            solver.Add(np.sum(x[i]) <= 1)
            
        # Constraints: UAV j has at most capacities[j] connections
        for j in range(self.n_tx):
            solver.Add(np.sum(x[:, j]) <= capacities[j])

        # Adding objective
        objective = solver.Objective()
        for i in range(self.n_rx):
            for j in range(self.n_tx):
                objective.SetCoefficient(x[i][j], scores[i][j])
        objective.SetMaximization()

        # Solving
        if solver.Solve() == pywraplp.Solver.OPTIMAL:
            rtn = []
            for j in range(self.n_tx):
                rtn.append([i for i in range(self.n_rx) if x[i][j].solution_value() > 0.5])
            coverage = sum([len(x) for x in rtn]) / self.n_rx
            return rtn, objective.Value(), coverage
        else:
            raise ValueError("The model doesn't converge for this input")