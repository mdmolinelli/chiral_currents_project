import numpy as np
import pulp


class CrosstalkCompensationOptimization:

    def __init__(self, crosstalk_matrix, crosstalk_inverse_matrix, crosstalk_offset_vector, voltage_threshold=8,
                 single_channel_threshold=4):

        self.crosstalk_matrix = crosstalk_matrix
        self.crosstalk_inverse_matrix = crosstalk_inverse_matrix
        self.crosstalk_offset_vector = crosstalk_offset_vector

        self.voltage_threshold = voltage_threshold
        self.single_channel_threshold = single_channel_threshold


    def integer_programming_get_combination(self, initial_fluxes, single_channel_threshold=2, fixed_indices=[]):
        # Create the problem instance
        problem = pulp.LpProblem('Minimize_L1_Norm', pulp.LpMinimize)

        d_vars = []

        for i in range(len(initial_fluxes)):
            if i in fixed_indices:
                d_vars.append(0)
            else:
                d_vars.append(pulp.LpVariable(f"d_{i}", lowBound=-3, upBound=3, cat='Integer'))


        # Calculate the voltage vector after transformation
        voltage_vector = self.flux_to_voltage(initial_fluxes + np.array(d_vars))

        # Define auxiliary variables for absolute values
        abs_vars = [pulp.LpVariable(f"abs_{i}", lowBound=0) for i in range(len(voltage_vector))]

        # Add constraints to ensure abs_vars[i] >= voltage_vector[i] and abs_vars[i] >= -voltage_vector[i]
        for i in range(len(voltage_vector)):
            problem += abs_vars[i] >= voltage_vector[i]
            problem += abs_vars[i] >= -voltage_vector[i]
            # Add constraints to ensure abs_vars[i] <= single_channel_threshold
            problem += abs_vars[i] <= single_channel_threshold

        # Objective function: Minimize the sum of absolute values (L1 norm)
        l1_norm = pulp.lpSum(abs_vars)
        problem += l1_norm

        # Solve the problem
        problem.solve(pulp.PULP_CBC_CMD(msg=False))

        # Extract the optimal adjustments
        optimal_adjustments = [pulp.value(d_var) for d_var in d_vars]

        return np.array(optimal_adjustments)

    def integer_programming(self, initial_fluxes, single_channel_threshold=2, fixed_indices=[]):
        optimal_adjustments = self.integer_programming_get_combination(initial_fluxes,
                                                                       single_channel_threshold=single_channel_threshold,
                                                                       fixed_indices=fixed_indices)
        optimal_fluxes = initial_fluxes + optimal_adjustments
        voltages = self.flux_to_voltage(optimal_fluxes)
        return voltages

    def flux_to_voltage(self, fluxes):
        fluxes = np.copy(np.array(fluxes))
        return np.dot(self.crosstalk_inverse_matrix, fluxes + self.crosstalk_offset_vector)
