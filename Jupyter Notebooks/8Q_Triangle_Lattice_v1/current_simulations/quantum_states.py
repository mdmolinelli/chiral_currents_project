import numpy as np



class Operator:
    def __init__(self, state, basis):
        '''
        :param state: list of complex numbers representing operator matrix elements
        :param basis: list of FockBasisOuter objects representing the basis states
        '''
        self.state = np.array(state)
        self.basis = basis

        if len(basis) == 0:
            self.basis = [FockBasisOuter(None, None)]
            self.state = np.array([0])
        self.num_qubits = self.basis[0].num_qubits

        self.basis_to_index = {}

        for i in range(len(self.basis)):
            self.basis_to_index[self.basis[i]] = i

    def trace(self):
        '''Return trace of density matrix (should be 1 for normalized states)'''
        trace_sum = 0
        for i in range(len(self.basis)):
            if self.basis[i].trace() == 1:
                trace_sum += self.state[i]
        return trace_sum
    
    def adjoint(self):
        new_state = []
        new_basis = []
        for i in range(len(self.basis)):
            new_basis.append(self.basis[i].adjoint())
            new_state.append(np.conjugate(self.state[i]))

        return Operator(new_state, new_basis)

    def generate_zero_operator(self):
        return Operator(np.array([0]), [FockBasisOuter(None, None)])

    def apply_raising_operator(self, qubit_index, num_levels):
        new_state = []
        new_basis = []
        for i in range(len(self.state)):
            state_number = self.basis[i].ket_state.state[qubit_index]
            new_basis_state = self.basis[i].apply_raising_operator(qubit_index, num_levels)
            if not new_basis_state.zero:
                new_basis.append(new_basis_state)
                new_state.append(np.sqrt(state_number+1)*self.state[i])

        return Operator(new_state, new_basis)
    
    def apply_raising_operator_left(self, qubit_index, num_levels):
        return self.apply_raising_operator(qubit_index, num_levels)

    def apply_raising_operator_right(self, qubit_index, num_levels):
        new_state = []
        new_basis = []
        for i in range(len(self.state)):
            state_number = self.basis[i].bra_state.state[qubit_index]
            new_basis_state = self.basis[i].apply_raising_operator_right(qubit_index)
            if not new_basis_state.zero:
                new_basis.append(new_basis_state)
                new_state.append(np.sqrt(state_number)*self.state[i])

        return Operator(new_state, new_basis)

    def apply_lowering_operator(self, qubit_index):
        new_state = []
        new_basis = []
        for i in range(len(self.state)):
            state_number = self.basis[i].ket_state.state[qubit_index]
            new_basis_state = self.basis[i].apply_lowering_operator(qubit_index)
            if not new_basis_state.zero:
                new_basis.append(new_basis_state)
                new_state.append(np.sqrt(state_number)*self.state[i])

        return Operator(new_state, new_basis)

    def apply_lowering_operator_left(self, qubit_index):
        return self.apply_lowering_operator(qubit_index)

    def apply_lowering_operator_right(self, qubit_index, num_levels):
        new_state = []
        new_basis = []
        for i in range(len(self.state)):
            state_number = self.basis[i].bra_state.state[qubit_index]
            new_basis_state = self.basis[i].apply_lowering_operator_right(qubit_index, num_levels)
            if not new_basis_state.zero:
                new_basis.append(new_basis_state)
                new_state.append(np.sqrt(state_number+1)*self.state[i])

        return Operator(new_state, new_basis)
        

    def __str__(self):
        result = "Operator:\n"
        for i in range(len(self.basis)):
            if abs(self.state[i]) > 1e-10:
                result += f"O[{self.basis[i]}] = {self.state[i]:.6f}\n"
        return result
    
    def __repr__(self):
        return f"Operator(shape={self.state.shape}, basis_size={len(self.basis)})"
    
    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            return Operator(other * self.state, self.basis)
        elif isinstance(other, Operator):
            new_basis = []
            new_state = []
            for i in range(len(self.basis)):
                for j in range(len(other.basis)):
                    new_basis_state = self.basis[i] * other.basis[j]
                    if not new_basis_state.zero:
                        new_basis.append(new_basis_state)
                        new_state.append(self.state[i] * other.state[j])
            return Operator(np.array(new_state), new_basis)
        else:
            raise TypeError("Cannot multiply Operator with type {}".format(type(other)))
        
    def __rmul__(self, other):
        if isinstance(other, (int, float, complex)):
            return Operator(other * self.state, self.basis)
        elif isinstance(other, Operator):
            new_basis = []
            new_state = []
            for i in range(len(self.basis)):
                for j in range(len(other.basis)):
                    new_basis_state = other.basis[i] * self.basis[j]
                    if not new_basis_state.zero:
                        new_basis.append(new_basis_state)
                        new_state.append(self.state[i] * other.state[j])
            return Operator(np.array(new_state), new_basis)
        else:
            raise TypeError("Cannot multiply Operator with type {}".format(type(other)))
        
    
    def __add__(self, other):
        if not isinstance(other, Operator):
            raise TypeError("Can only add Operator objects.")

        new_basis = list(self.basis.copy())
        new_state = list(self.state.copy())
        
        for key in other.basis_to_index:
            if key in self.basis_to_index:
                new_state[self.basis_to_index[key]] += other.state[other.basis_to_index[key]]
            else:
                new_basis.append(key)
                new_state.append(other.state[other.basis_to_index[key]])
       
        return Operator(new_state, new_basis)

    def __eq__(self, other):
        if not isinstance(other, Operator):
            return False
        
        for key in self.basis_to_index:
            if key not in other.basis_to_index:
                return False
            if not np.isclose(self.state[self.basis_to_index[key]], other.state[other.basis_to_index[key]]):
                return False
        
        return True

class DensityMatrix(Operator):
    def __init__(self, state, basis):
        super().__init__(state, basis)

        if not np.isclose(self.trace(), 1):
            raise ValueError("Density matrix must have trace 1.")
        
        if not (self == self.adjoint()):
            raise ValueError("Density matrix must be Hermitian.")

    def purity(self):
        """
        Highly optimized purity calculation: Tr(ρ²) = Σᵢⱼ |ρᵢⱼ|²
        
        For a density matrix ρ = Σᵢⱼ ρᵢⱼ |i⟩⟨j|, the purity is simply 
        the sum of the squared magnitudes of all matrix elements.
        """
        # Group coefficients by (bra, ket) pairs to handle potential duplicates
        element_dict = {}
        
        for i, basis_elem in enumerate(self.basis):
            if basis_elem.zero:
                continue
                
            key = (tuple(basis_elem.bra_state.state) if basis_elem.bra_state.state is not None else None,
                   tuple(basis_elem.ket_state.state) if basis_elem.ket_state.state is not None else None)
            
            if key in element_dict:
                element_dict[key] += self.state[i]
            else:
                element_dict[key] = self.state[i]
        
        # Purity is sum of |ρᵢⱼ|² for all matrix elements
        purity_sum = sum(abs(coeff)**2 for coeff in element_dict.values())
        
        return purity_sum

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

        if isinstance(state, FockBasisState):
            self.state = state.state
        else:
            self.state = state



        self.zero = False



        if self.state is None:
            self.zero = True
            self.num_qubits = 0
        else:
            self.state = np.array(self.state)
            self.num_qubits = len(self.state)



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

    def __hash__(self):
        if self.zero:
            return hash(None)
        return hash(tuple(self.state))

    def __str__(self):
        return str(self.state)
    
    def __repr__(self):
        return str(self.state)
    
class FockBasisOuter:
    '''
    Class to represent outer product of two Fock basis states |a><b|
    '''

    def __init__(self, ket_state, bra_state):
        '''
        :param bra_state: list or FockBasisState object
        :param ket_state: list or FockBasisState object
        '''

        if isinstance(ket_state, (list, tuple, np.ndarray)):
            ket_state = FockBasisState(ket_state)
            
        if isinstance(bra_state, (list, tuple, np.ndarray)):
            bra_state = FockBasisState(bra_state)
        

        self.ket_state = ket_state
        self.bra_state = bra_state

        self.zero = False

        if not ket_state is None and not bra_state is None:
            if not ket_state.state is None and not bra_state.state is None:
                self.num_qubits = len(ket_state.state)

                if not len(ket_state.state) == len(bra_state.state):
                    raise ValueError("Ket and bra states must have the same number of qubits.")
            else:
                self.zero = True
                self.num_qubits = 0
        else:
            self.zero = True
            self.num_qubits = 0


    def trace(self):
        if self.bra_state == self.ket_state:
            return 1
        else:
            return 0

    def adjoint(self):
        return FockBasisOuter(self.bra_state, self.ket_state)

    def apply_raising_operator(self, qubit_index, num_levels):
        new_ket_state = self.ket_state.apply_raising_operator(qubit_index, num_levels)
        return FockBasisOuter(new_ket_state, self.bra_state)

    def apply_raising_operator_left(self, qubit_index, num_levels):
        return self.apply_raising_operator(qubit_index, num_levels)
    
    def apply_raising_operator_right(self, qubit_index):
        '''
        Equivalent to applying lowering operator to bra state
        '''
        new_bra_state = self.bra_state.apply_lowering_operator(qubit_index)
        return FockBasisOuter(self.ket_state, new_bra_state)


    def apply_lowering_operator(self, qubit_index):
        new_ket_state = self.ket_state.apply_lowering_operator(qubit_index)
        return FockBasisOuter(new_ket_state, self.bra_state)
    
    def apply_lowering_operator_left(self, qubit_index):
        return self.apply_lowering_operator(qubit_index)

    def apply_lowering_operator_right(self, qubit_index, num_levels):
        new_bra_state = self.bra_state.apply_raising_operator(qubit_index, num_levels)
        return FockBasisOuter(self.ket_state, new_bra_state)
    

    def generate_zero_state(self):
        return FockBasisOuter(None, None)
    
    def __eq__(self, other):
        return np.all(self.ket_state == other.ket_state) and np.all(self.bra_state == other.bra_state)

    def __str__(self):
        return str(self.ket_state) + ", " + str(self.bra_state)
    
    def __repr__(self):
        return str(self)

    def __mul__(self, other):
        if np.all(self.bra_state == other.ket_state):
            return FockBasisOuter(self.ket_state, other.bra_state)
        else:
            return FockBasisOuter(None, None)

    def __rmul__(self, other):
        if np.all(other.bra_state == self.ket_state):
            return FockBasisOuter(other.ket_state, self.bra_state)
        else:
            return FockBasisOuter(None, None)

    def __hash__(self):
        return hash((tuple(self.ket_state.state) if not self.ket_state.zero else None,
                     tuple(self.bra_state.state) if not self.bra_state.zero else None))

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

    ### FockBasisOuter test
    print('='*40)
    print('Testing FockBasisOuter class:')


    outer = FockBasisOuter(FockBasisState([0,1]), [0,1])
    print("Trace of outer product (should be 1):", outer.trace())  # Expected: 1
    outer2 = FockBasisOuter(FockBasisState([0,1]), FockBasisState([1,0]))
    print("Trace of outer product (should be 0):", outer2.trace())  # Expected: 0

    print(f'test raising and lowering operators, starting from:')
    print(outer2)

    adjoint_outer = outer2.adjoint()
    print("Adjoint of outer product:", adjoint_outer)  # Expected: |1,0><0,1|

    raised_outer = outer2.apply_raising_operator(1, 3)
    print("Raised outer product left (qubit 1):", raised_outer)  # Expected |0,2><1,0|


    raised_outer_right = outer2.apply_raising_operator_right(0)
    print("Raised outer product right (qubit 0):", raised_outer_right)  # Expected |0,1><0,0|

    lowered_outer = outer2.apply_lowering_operator(1)
    print("Lowered outer product left (qubit 1):", lowered_outer)  # Expected |0,0><0,1|

    lowered_outer = outer2.apply_lowering_operator_right(1, 3)
    print("Lowered outer product right (qubit 1):", lowered_outer)  # Expected |0,0><1,1|

    print('='*50)
    print('Testing Operator class:')
    print('='*50)

    # Test basic operator construction
    basis_op = [
        FockBasisOuter(FockBasisState([0, 0]), FockBasisState([0, 0])),  # |00⟩⟨00|
        FockBasisOuter(FockBasisState([1, 0]), FockBasisState([1, 0])),  # |10⟩⟨10|
        FockBasisOuter(FockBasisState([0, 0]), FockBasisState([1, 0])),  # |00⟩⟨10|
    ]
    state_op = [0.5, 0.3, 0.2j]
    operator = Operator(state_op, basis_op)
    
    print("Operator created successfully")
    print("Operator trace:", operator.trace())  # Should be 0.8 (only diagonal terms)
    print("Operator representation:", repr(operator))

    # Test operator arithmetic
    scaled_op = 2.0 * operator
    print("Scaled operator trace:", scaled_op.trace())  # Should be 1.6

    # Test operator addition
    basis_op2 = [
        FockBasisOuter(FockBasisState([0, 0]), FockBasisState([0, 0])),  # |00⟩⟨00|
        FockBasisOuter(FockBasisState([0, 0]), FockBasisState([1, 0])),  # |00⟩⟨10|
        FockBasisOuter(FockBasisState([1, 1]), FockBasisState([1, 0])),  # |00⟩⟨10|
    ]
    state_op2 = [0.2, 0.1, -0.1j, 0.2]
    operator2 = Operator(state_op2, basis_op2)
    
    sum_op = operator + operator2

    print(f'testing addition:')
    print(f'{operator} + {operator2} = {sum_op}')

    print("Sum operator trace:", sum_op.trace())  # Should be 1.1

    # Test operator equality
    operator3 = Operator(state_op.copy(), basis_op.copy())
    print("Operator equality test:", operator == operator3)  # Should be True
    print("Operator inequality test:", operator == operator2)  # Should be False

    # Test operator multiplication (this might fail due to implementation issues)
    try:
        mult_op = operator * operator2
        print("Operator multiplication successful")
        print("Product trace:", mult_op.trace())
    except Exception as e:
        print(f"Operator multiplication failed: {e}")

    # Test adjoint
    try:
        adj_op = operator.adjoint()
        if adj_op is not None:
            print("Adjoint operator created")
    except Exception as e:
        print(f"Adjoint operation failed: {e}")

    print('='*50)
    print('Testing DensityMatrix class:')
    print('='*50)

    # Create a valid density matrix (normalized, Hermitian)
    # Simple 2-level system: 0.6|00⟩⟨00| + 0.4|10⟩⟨10|
    basis_dm = [
        FockBasisOuter(FockBasisState([0, 0]), FockBasisState([0, 0])),  # |00⟩⟨00|
        FockBasisOuter(FockBasisState([1, 0]), FockBasisState([1, 0])),  # |10⟩⟨10|
    ]
    state_dm = [0.6, 0.4]  # Probabilities sum to 1
    
    try:
        density_matrix = DensityMatrix(state_dm, basis_dm)
        print("Density matrix created successfully")
        print("Density matrix trace:", density_matrix.trace())  # Should be 1.0
        print("Density matrix purity:", density_matrix.purity())  # Should be 0.52
    except Exception as e:
        print(f"Density matrix creation failed: {e}")

    # Test with invalid density matrix (trace ≠ 1)
    try:
        invalid_dm = DensityMatrix([0.3, 0.2], basis_dm)
        print("Invalid density matrix created (this shouldn't happen)")
    except ValueError as e:
        print(f"Correctly caught invalid trace: {e}")

    # Test with non-Hermitian density matrix
    try:
        non_hermitian_basis = [
            FockBasisOuter(FockBasisState([0, 0]), FockBasisState([0, 0])),  # |00⟩⟨00|
            FockBasisOuter(FockBasisState([1, 0]), FockBasisState([1, 0])),  # |10⟩⟨10|
            FockBasisOuter(FockBasisState([0, 0]), FockBasisState([1, 0])),  # |00⟩⟨10|
        ]
        non_hermitian_state = [0.5, 0.5, 0.2]  # Not Hermitian: missing |10⟩⟨00| conjugate
        non_herm_dm = DensityMatrix(non_hermitian_state, non_hermitian_basis)
        print("Non-Hermitian density matrix created (this shouldn't happen)")
    except ValueError as e:
        print(f"Correctly caught non-Hermitian matrix: {e}")

    # Test mixed state density matrix with off-diagonal terms
    basis_dm_mixed = [
        FockBasisOuter(FockBasisState([0, 0]), FockBasisState([0, 0])),  # |00⟩⟨00|
        FockBasisOuter(FockBasisState([1, 0]), FockBasisState([1, 0])),  # |10⟩⟨10|
        FockBasisOuter(FockBasisState([0, 0]), FockBasisState([1, 0])),  # |00⟩⟨10|
        FockBasisOuter(FockBasisState([1, 0]), FockBasisState([0, 0])),  # |10⟩⟨00|
    ]
    
    # Coherent superposition: (|00⟩ + |10⟩)/√2
    # ρ = 0.5|00⟩⟨00| + 0.5|10⟩⟨10| + 0.5|00⟩⟨10| + 0.5|10⟩⟨00|
    state_dm_mixed = [0.5, 0.5, 0.5, 0.5]
    
    try:
        mixed_dm = DensityMatrix(state_dm_mixed, basis_dm_mixed)
        print("Mixed density matrix created successfully")
        print("Mixed density matrix trace:", mixed_dm.trace())  # Should be 1.0
        print("Mixed density matrix purity:", mixed_dm.purity())  # Should be 1.0 (pure state)
    except Exception as e:
        print(f"Mixed density matrix creation failed: {e}")

    # Test density matrix arithmetic
    if 'density_matrix' in locals():
        try:
            scaled_dm = 0.5 * density_matrix
            print("Scaled density matrix trace:", scaled_dm.trace())  # Should be 0.5
            
            # This should fail since it won't have trace 1
            try:
                invalid_scaled_dm = DensityMatrix(scaled_dm.state, scaled_dm.basis)
                print("Scaled density matrix incorrectly accepted")
            except ValueError as e:
                print(f"Correctly rejected scaled density matrix: {e}")
                
        except Exception as e:
            print(f"Density matrix arithmetic failed: {e}")

    # Test pure state density matrix
    # Create |01⟩⟨01| (pure state)
    basis_pure = [FockBasisOuter(FockBasisState([0, 1]), FockBasisState([0, 1]))]
    state_pure = [1.0]
    
    try:
        pure_dm = DensityMatrix(state_pure, basis_pure)
        print("Pure state density matrix created")
        print("Pure state trace:", pure_dm.trace())
        print("Pure state purity:", pure_dm.purity())  # Should be 1.0
    except Exception as e:
        print(f"Pure state density matrix failed: {e}")

    print("\nAll Operator and DensityMatrix tests completed!")


    print(f'testing trace measurment:')
    print((mixed_dm*operator2).trace())
    print((operator2*mixed_dm).trace())


