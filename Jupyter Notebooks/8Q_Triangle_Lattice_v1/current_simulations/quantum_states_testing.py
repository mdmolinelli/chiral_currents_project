import numpy as np
import pytest

from quantum_states import (
    Operator, DensityMatrix, QuantumState, FockBasisState, FockBasisOuter
)


class TestFockBasisState:
    def test_initialization(self):
        state = FockBasisState([0, 1, 2])
        assert np.array_equal(state.state, np.array([0, 1, 2]))
        assert state.num_qubits == 3
        assert not state.zero

    def test_zero_state(self):
        state = FockBasisState(None)
        assert state.zero
        assert state.num_qubits == 0

    def test_equality(self):
        state1 = FockBasisState([0, 1, 2])
        state2 = FockBasisState([0, 1, 2])
        state3 = FockBasisState([1, 0, 2])
        assert state1 == state2
        assert not (state1 == state3)

    def test_inner_product(self):
        state1 = FockBasisState([0, 1, 2])
        state2 = FockBasisState([0, 1, 2])
        state3 = FockBasisState([1, 0, 2])
        assert state1.inner_product(state2) == 1
        assert state1.inner_product(state3) == 0

    def test_raising_operator(self):
        state = FockBasisState([0, 1, 2])
        raised = state.apply_raising_operator(1, 3)
        assert np.array_equal(raised.state, np.array([0, 2, 2]))
        
    def test_raising_operator_exceeds_levels(self):
        state = FockBasisState([0, 2, 2])
        raised = state.apply_raising_operator(1, 3)
        assert raised.zero

    def test_lowering_operator(self):
        state = FockBasisState([0, 1, 2])
        lowered = state.apply_lowering_operator(2)
        assert np.array_equal(lowered.state, np.array([0, 1, 1]))

    def test_lowering_operator_zero(self):
        state = FockBasisState([0, 0, 0])
        lowered = state.apply_lowering_operator(0)
        assert lowered.zero


class TestQuantumState:
    def test_initialization(self):
        basis = [FockBasisState([0, 0]), FockBasisState([1, 0])]
        state = [1 + 1j, 2 + 0j]
        qs = QuantumState(state, basis)
        assert qs.num_qubits == 2
        assert len(qs.basis) == 2

    def test_zero_state(self):
        qs = QuantumState([0], [FockBasisState(None)])
        assert qs.num_qubits == 0

    def test_norm(self):
        basis = [FockBasisState([0, 0]), FockBasisState([1, 0])]
        state = [1 + 1j, 2 + 0j]
        qs = QuantumState(state, basis)
        expected_norm = np.sqrt(abs(1+1j)**2 + abs(2+0j)**2)
        assert np.isclose(qs.norm(), expected_norm)

    def test_inner_product(self):
        basis = [FockBasisState([0, 0]), FockBasisState([1, 0])]
        state1 = [1 + 1j, 2 + 0j]
        state2 = [1 - 1j, 2 + 0j]
        qs1 = QuantumState(state1, basis)
        qs2 = QuantumState(state2, basis)
        inner = qs1.inner_product(qs2)
        expected = np.conj(1+1j) * (1-1j) + np.conj(2+0j) * (2+0j)
        assert np.isclose(inner, expected)

    def test_raising_operator(self):
        basis = [FockBasisState([0, 0])]
        qs = QuantumState([1.0], basis)
        raised = qs.apply_raising_operator(0, 3)
        assert raised.basis[0] == FockBasisState([1, 0])
        assert np.isclose(raised.state[0], 1.0)

    def test_lowering_operator(self):
        basis = [FockBasisState([1, 0])]
        qs = QuantumState([1.0], basis)
        lowered = qs.apply_lowering_operator(0)
        assert lowered.basis[0] == FockBasisState([0, 0])

    def test_addition(self):
        basis1 = [FockBasisState([0, 0]), FockBasisState([1, 0])]
        basis2 = [FockBasisState([0, 0]), FockBasisState([1, 0])]
        qs1 = QuantumState([1.0, 2.0], basis1)
        qs2 = QuantumState([0.5, 0.5], basis2)
        added = qs1 + qs2
        assert np.isclose(added.state[0], 1.5)
        assert np.isclose(added.state[1], 2.5)

    def test_scalar_multiplication(self):
        basis = [FockBasisState([0, 0])]
        qs = QuantumState([1.0 + 1.0j], basis)
        scaled = 2.0 * qs
        assert np.isclose(scaled.state[0], 2.0 + 2.0j)


class TestFockBasisOuter:
    def test_initialization(self):
        outer = FockBasisOuter([0, 1], [0, 1])
        assert outer.num_qubits == 2
        assert not outer.zero

    def test_trace_same_states(self):
        outer = FockBasisOuter([0, 1], [0, 1])
        assert outer.trace() == 1

    def test_trace_different_states(self):
        outer = FockBasisOuter([0, 1], [1, 0])
        assert outer.trace() == 0

    def test_adjoint(self):
        outer = FockBasisOuter([0, 1], [1, 0])
        adj = outer.adjoint()
        assert np.array_equal(adj.ket_state, FockBasisState([1, 0]))
        assert np.array_equal(adj.bra_state, FockBasisState([0, 1]))

    def test_multiplication(self):
        outer1 = FockBasisOuter([0, 1], [0, 1])
        outer2 = FockBasisOuter([0, 1], [1, 0])
        result = outer1 * outer2
        assert result.zero

    def test_raising_operator_left(self):
        outer = FockBasisOuter([0, 1], [1, 0])
        raised = outer.apply_raising_operator(1, 3)
        assert np.array_equal(raised.ket_state, FockBasisState([0, 2]))

    def test_raising_operator_right(self):
        outer = FockBasisOuter([0, 1], [1, 0])
        raised = outer.apply_raising_operator_right(0)
        assert np.array_equal(raised.bra_state, FockBasisState([0, 0]))

    def test_lowering_operator_left(self):
        outer = FockBasisOuter([1, 1], [1, 0])
        lowered = outer.apply_lowering_operator(0)
        assert np.array_equal(lowered.ket_state, FockBasisState([0, 1]))

    def test_lowering_operator_right(self):
        outer = FockBasisOuter([0, 1], [1, 0])
        lowered = outer.apply_lowering_operator_right(1, 3)
        assert np.array_equal(lowered.bra_state, FockBasisState([1, 1]))


class TestOperator:
    def test_initialization(self):
        basis = [FockBasisOuter([0], [0]), FockBasisOuter([1], [1])]
        state = np.array([1.0, 0.5])
        op = Operator(state, basis)
        assert op.num_qubits == 1
        assert len(op.basis) == 2

    def test_empty_initialization(self):
        op = Operator([], [])
        assert op.state[0] == 0
        assert op.basis[0].zero

    def test_trace(self):
        basis = [FockBasisOuter([0], [0]), FockBasisOuter([1], [1])]
        state = np.array([0.7, 0.3])
        op = Operator(state, basis)
        assert np.isclose(op.trace(), 1.0)

    def test_scalar_multiplication(self):
        basis = [FockBasisOuter([0], [0])]
        state = np.array([1.0])
        op = Operator(state, basis)
        scaled = 2.0 * op
        assert np.isclose(scaled.state[0], 2.0)

    def test_scalar_rmultiplication(self):
        basis = [FockBasisOuter([0], [0])]
        state = np.array([1.0])
        op = Operator(state, basis)
        scaled = op * 3.0
        assert np.isclose(scaled.state[0], 3.0)

    def test_addition(self):
        basis = [FockBasisOuter([0], [0]), FockBasisOuter([1], [1])]
        state1 = np.array([0.5, 0.5])
        state2 = np.array([0.3, 0.7])
        op1 = Operator(state1, basis)
        op2 = Operator(state2, basis)
        added = op1 + op2
        assert np.isclose(added.state[0], 0.8)
        assert np.isclose(added.state[1], 1.2)

    def test_equality(self):
        basis = [FockBasisOuter([0], [0])]
        state = np.array([1.0])
        op1 = Operator(state, basis)
        op2 = Operator(state, basis)
        assert op1 == op2

    def test_inequality_different_basis(self):
        basis1 = [FockBasisOuter([0], [0])]
        basis2 = [FockBasisOuter([1], [1])]
        state = np.array([1.0])
        op1 = Operator(state, basis1)
        op2 = Operator(state, basis2)
        assert not (op1 == op2)


class TestDensityMatrix:
    def test_initialization(self):
        basis = [FockBasisOuter([0], [0]), FockBasisOuter([1], [1])]
        state = np.array([0.7, 0.3])
        dm = DensityMatrix(state, basis)
        assert dm.num_qubits == 1
        assert np.isclose(dm.trace(), 1.0)

    def test_non_normalized_raises_error(self):
        basis = [FockBasisOuter([0], [0]), FockBasisOuter([1], [1])]
        state = np.array([0.5, 0.3])
        with pytest.raises(ValueError, match="trace 1"):
            DensityMatrix(state, basis)

    def test_purity(self):
        basis = [FockBasisOuter([0], [0]), FockBasisOuter([1], [1])]
        state = np.array([1.0, 0.0])
        dm = DensityMatrix(state, basis)
        assert np.isclose(dm.purity(), 1.0)

    def test_purity_mixed_state(self):
        basis = [FockBasisOuter([0], [0]), FockBasisOuter([1], [1])]
        state = np.array([0.5, 0.5])
        dm = DensityMatrix(state, basis)
        assert dm.purity() < 1.0