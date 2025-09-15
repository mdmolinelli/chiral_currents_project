import numpy as np

class QuantumState:
    def __init__(self, state, basis):
        '''
        :param data: list of complex numbers
        :param basis: list of FockBasisState objects
        '''
        self.state = np.array(state)
        self.basis = basis


        if len(basis) == 0:
            self.basis = [FockBasisState(None)]
            self.state = [0]
        self.num_qubits = self.basis[0].num_qubits

        self.tuple_to_coeff = {}

        for i in range(len(self.basis)):
            if self.basis[i].zero:
                continue
            self.tuple_to_coeff[tuple(self.basis[i].state)] = self.state[i]


    def norm(self):
        return np.linalg.norm(self.state)
        
    def inner_product(self, other_state):
        '''
        This object is the bra (adjoint), other_state is the ket
        '''
        value = 0

        if self.basis[0].zero or other_state.basis[0].zero:
            return 0
        
        for i in range(len(self.state)):

            
            basis_state_tuple = tuple(self.basis[i].state)
            if basis_state_tuple in other_state.tuple_to_coeff:
                coeff = self.state[i]
                other_state_coeff = other_state.tuple_to_coeff[basis_state_tuple]
                value += np.conj(coeff) * other_state_coeff
        return value
    
    def apply_raising_operator(self, qubit_index, num_levels):
        new_state = []
        new_basis = []
        for i in range(len(self.state)):
            state_number = self.basis[i].state[qubit_index]
            new_basis_state = self.basis[i].apply_raising_operator(qubit_index, num_levels)
            if not new_basis_state.zero:
                new_basis.append(new_basis_state)
                new_state.append(np.sqrt(state_number+1)*self.state[i])

        return QuantumState(new_state, new_basis)
    
    def apply_lowering_operator(self, qubit_index):
        new_state = []
        new_basis = []
        for i in range(len(self.state)):
            state_number = 0
            if not self.basis[i].zero:
                state_number = self.basis[i].state[qubit_index]
            new_basis_state = self.basis[i].apply_lowering_operator(qubit_index)
            if not new_basis_state.zero:
                new_basis.append(new_basis_state)
                new_state.append(np.sqrt(state_number)*self.state[i])

        if len(new_basis) == 0:
            return self.generate_zero_state()

        return QuantumState(new_state, new_basis)
    
    def generate_zero_state(self):
        return QuantumState([0], [FockBasisState(None)])

    def __str__(self):
        return ", ".join(f"{self.state[i]}({self.basis[i]})" for i in range(len(self.state)))
    
    def __repr__(self):
        return ", ".join(f"{self.state[i]}({self.basis[i]})" for i in range(len(self.state)))
    
    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            new_state = other*self.state
            return QuantumState(new_state, self.basis)
        elif isinstance(other, QuantumState):
            raise TypeError("Cannot multiply two QuantumState objects.")
        
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __add__(self, other):
        if not isinstance(other, QuantumState):
            raise TypeError("Can only add QuantumState objects.")
        
        new_state_basis = []
        new_state = []

        for i in range(len(self.state)):
            new_state.append(self.state[i])
            new_state_basis.append(self.basis[i])
        for i in range(len(other.state)):
            if other.basis[i] in new_state_basis:
                index = new_state_basis.index(other.basis[i])
                new_state[index] += other.state[i]
            else:
                new_state.append(other.state[i])
                new_state_basis.append(other.basis[i])
        return QuantumState(new_state, new_state_basis)

class FockBasisState:
    def __init__(self, state):
        '''
        :param data: list of integers
        '''

        self.zero = False
        self.state = np.array(state)

        if state is None:
            self.zero = True
            self.num_qubits = 0
        else:
            self.num_qubits = len(state)


    def inner_product(self, other_state):
        if self == other_state:
            return 1
        return 0
    
    def apply_raising_operator(self, qubit_index, num_levels):
        if self.zero:
            return self.generate_zero_state()
        new_data = self.state.copy()
        new_data[qubit_index] += 1
        if new_data[qubit_index] >= num_levels:
            return self.generate_zero_state()
        new_state = FockBasisState(new_data)
        return new_state
        
    def apply_lowering_operator(self, qubit_index):
        if self.zero or self.state[qubit_index] == 0:
            return self.generate_zero_state()
        new_data = self.state.copy()
        new_data[qubit_index] -= 1
        new_state = FockBasisState(new_data)
        return new_state
    
    def generate_zero_state(self):
        return FockBasisState(None)
    
    def __eq__(self, other):
        return np.all(self.state == other.state)

    def __str__(self):
        return str(self.state)
    
    def __repr__(self):
        return str(self.state)

if __name__ == "__main__":
    # Test the QuantumState and FockBasisState classes

    # Test FockBasisState
    state1 = FockBasisState([0, 1, 2])
    state2 = FockBasisState([0, 1, 2])
    state3 = FockBasisState([1, 0, 2])

    # Test inner product
    print("Inner product (state1, state2):", state1.inner_product(state2))  # Expected: 1
    print("Inner product (state1, state3):", state1.inner_product(state3))  # Expected: 0

    # Test raising operator
    raised_state = state1.apply_raising_operator(1, 3)
    print("Raised state (qubit 1):", raised_state)  # Expected: [0, 2, 2]

    # Test lowering operator
    lowered_state = state1.apply_lowering_operator(2)
    print("Lowered state (qubit 2):", lowered_state)  # Expected: [0, 1, 1]

    # Test zero state generation
    zero_state = state1.generate_zero_state()
    print("Zero state:", zero_state)  # Expected: None

    # Test QuantumState
    basis = [FockBasisState([0, 0]), FockBasisState([1, 0])]
    state = [1 + 1j, 2 + 0j]
    quantum_state = QuantumState(state, basis)

    print(quantum_state.tuple_to_coeff)

    # Test norm
    print("Norm of quantum state:", quantum_state.norm())  # Expected: sqrt(6)

    # Test inner product
    other_basis = [FockBasisState([0, 0]), FockBasisState([1, 0])]
    other_state = [1 - 1j, 2 + 0j]
    other_quantum_state = QuantumState(other_state, other_basis)
    print("Inner product of quantum states:", quantum_state.inner_product(other_quantum_state))  # Expected: Complex value

    # Test raising operator
    raised_quantum_state = quantum_state.apply_raising_operator(0, 3)
    print("Raised quantum state:", raised_quantum_state)

    raised_quantum_state = quantum_state.apply_raising_operator(0, 3).apply_raising_operator(0, 3)
    print("Raised quantum state:", raised_quantum_state)

    # Test lowering operator
    lowered_quantum_state = quantum_state.apply_lowering_operator(0)
    print("Lowered quantum state:", lowered_quantum_state)

    lowered_quantum_state = quantum_state.apply_lowering_operator(1)
    print("Lowered quantum state:", lowered_quantum_state)

    # Test addition
    added_quantum_state = quantum_state + other_quantum_state
    print("Quantum state:", quantum_state)
    print("Other quantum state:", other_quantum_state)
    print("Added quantum state:", added_quantum_state)

    # Test inner product with zero state
    zero_basis = [FockBasisState(None)]
    zero_state = [0]
    zero_quantum_state = QuantumState(zero_state, zero_basis)
    print("Inner product with zero state:", quantum_state.inner_product(zero_quantum_state))  # Expected: 0
    print("Inner product with zero state:", zero_quantum_state.inner_product(quantum_state))  # Expected: 0