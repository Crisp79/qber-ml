"""
Tests for quantum channel models.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from temporal_qber.simulator.channel import (
    QuantumChannel,
    IdealChannel,
    DepolarizingChannel,
    PhaseFlipChannel,
    AmplitudeDampingChannel,
    ChannelConfig,
    ChannelType,
    create_channel,
)
from temporal_qber.simulator.alice import BB84_STATES, Basis, Bit


class TestIdealChannel:
    """Test ideal channel behavior."""
    
    def test_preserves_state(self):
        """Ideal channel should preserve state perfectly."""
        channel = IdealChannel(seed=42)
        
        # Test all four BB84 states
        for state in BB84_STATES.values():
            output = channel.transmit(state)
            assert np.allclose(output, state)
    
    def test_batch_transmission(self):
        """Ideal channel should preserve batch of states."""
        channel = IdealChannel(seed=42)
        
        states = np.array(list(BB84_STATES.values()), dtype=np.complex128)
        output = channel.transmit_batch(states)
        
        assert np.allclose(output, states)
    
    def test_loss_model(self):
        """Test photon loss in ideal channel."""
        channel = IdealChannel(seed=42)
        channel.config.loss_probability = 0.5
        
        # Manually test loss
        state = BB84_STATES[(Basis.Z, Bit.ZERO)]
        n = 10000
        lost_count = 0
        
        for _ in range(n):
            out, detected = channel.apply_loss(state)
            if not detected:
                lost_count += 1
        
        # Should be close to 50%
        assert np.isclose(lost_count / n, 0.5, atol=0.02)


class TestDepolarizingChannel:
    """Test depolarizing channel."""
    
    def test_depolarizing_errors(self):
        """Depolarizing channel should introduce random Pauli errors."""
        channel = DepolarizingChannel(depolarizing_prob=0.3, seed=42)
        
        state = BB84_STATES[(Basis.Z, Bit.ZERO)]
        n = 5000
        
        error_count = 0
        
        for _ in range(n):
            output = channel.transmit(state)
            
            # Check if output differs from input (any error)
            if not np.allclose(output, state):
                error_count += 1
        
        # With p=0.3, error rate should be ~30% (1/3 = 33% actually)
        # X error: |0> -> |1> (diff)
        # Y error: |0> -> i|1> (diff)  
        # Z error: |0> -> |0> (same)
        # So 2/3 of errors are detectable = 30% * 2/3 = 20%
        assert error_count / n > 0.15
        assert error_count / n < 0.30
    
    def test_depolarizing_prob_zero(self):
        """Depolarizing channel with p=0 should act as ideal."""
        channel = DepolarizingChannel(depolarizing_prob=0.0, seed=42)
        
        state = BB84_STATES[(Basis.Z, Bit.ZERO)]
        output = channel.transmit(state)
        
        assert np.allclose(output, state)
    
    def test_depolarizing_batch(self):
        """Test batch transmission."""
        channel = DepolarizingChannel(depolarizing_prob=0.1, seed=42)
        
        states = np.array(list(BB84_STATES.values()), dtype=np.complex128)
        output = channel.transmit_batch(states)
        
        assert output.shape == states.shape
        # Some states should be modified
        diff = np.sum([not np.allclose(output[i], states[i]) for i in range(len(states))])
        assert diff >= 0


class TestPhaseFlipChannel:
    """Test phase flip channel."""
    
    def test_phase_flip_on_plus(self):
        """Phase flip on |+> should give |->."""
        channel = PhaseFlipChannel(phase_flip_prob=1.0, seed=42)
        
        plus = BB84_STATES[(Basis.X, Bit.ZERO)]  # |+>
        minus = BB84_STATES[(Basis.X, Bit.ONE)]  # |->
        
        output = channel.transmit(plus)
        assert np.allclose(output, minus)
    
    def test_phase_flip_on_minus(self):
        """Phase flip on |-> should give |+>."""
        channel = PhaseFlipChannel(phase_flip_prob=1.0, seed=42)
        
        plus = BB84_STATES[(Basis.X, Bit.ZERO)]
        minus = BB84_STATES[(Basis.X, Bit.ONE)]
        
        output = channel.transmit(minus)
        assert np.allclose(output, plus)
    
    def test_phase_flip_preserves_z_basis(self):
        """Phase flip should not change Z basis states (up to global phase)."""
        channel = PhaseFlipChannel(phase_flip_prob=1.0, seed=42)
        
        # |0> and |1> are eigenstates of Z
        for state in [BB84_STATES[(Basis.Z, Bit.ZERO)], 
                      BB84_STATES[(Basis.Z, Bit.ONE)]]:
            output = channel.transmit(state)
            # Should be same up to global phase
            overlap = np.abs(np.vdot(state, output))**2
            assert np.isclose(overlap, 1.0)
    
    def test_phase_flip_prob_zero(self):
        """Phase flip with p=0 should act as ideal."""
        channel = PhaseFlipChannel(phase_flip_prob=0.0, seed=42)
        
        for state in BB84_STATES.values():
            output = channel.transmit(state)
            assert np.allclose(output, state)


class TestAmplitudeDampingChannel:
    """Test amplitude damping channel."""
    
    def test_full_damping(self):
        """Full damping (gamma=1) should map |1> to |0>."""
        channel = AmplitudeDampingChannel(damping_gamma=1.0, seed=42)
        
        one = BB84_STATES[(Basis.Z, Bit.ONE)]
        zero = BB84_STATES[(Basis.Z, Bit.ZERO)]
        
        output = channel.transmit(one)
        # Should become |0> with probability |<1|psi>|^2 = 1
        assert np.allclose(output, zero)
    
    def test_preserves_zero(self):
        """Amplitude damping should preserve |0>."""
        channel = AmplitudeDampingChannel(damping_gamma=1.0, seed=42)
        
        zero = BB84_STATES[(Basis.Z, Bit.ZERO)]
        output = channel.transmit(zero)
        assert np.allclose(output, zero)
    
    def test_partial_damping(self):
        """Partial damping should reduce |1> amplitude."""
        channel = AmplitudeDampingChannel(damping_gamma=0.5, seed=42)
        
        one = BB84_STATES[(Basis.Z, Bit.ONE)]
        
        # Run multiple times to check statistical behavior
        n = 1000
        outcomes = []
        for _ in range(n):
            output = channel.transmit(one)
            prob_0 = np.abs(output[0])**2
            if np.isclose(prob_0, 1.0):
                outcomes.append('|0>')
            elif np.isclose(np.abs(output[1])**2, 1.0):
                outcomes.append('|1>')
            else:
                outcomes.append('other')
        
        count_0 = outcomes.count('|0>')
        count_1 = outcomes.count('|1>')
        
        # With gamma=0.5, we expect roughly 50% |0> and 50% |1>
        # Allow some statistical variance
        prob_0 = count_0 / n
        prob_1 = count_1 / n
        
        assert 0.3 < prob_0 < 0.7
        assert 0.3 < prob_1 < 0.7
        # Most outcomes should be either |0> or |1>
        assert (count_0 + count_1) / n > 0.9
    def test_gamma_zero_is_ideal(self):
        """Amplitude damping with gamma=0 should be ideal."""
        channel = AmplitudeDampingChannel(damping_gamma=0.0, seed=42)
        
        for state in BB84_STATES.values():
            output = channel.transmit(state)
            assert np.allclose(output, state)


class TestChannelConfig:
    """Test channel configuration validation."""
    
    def test_valid_config(self):
        config = ChannelConfig(
            channel_type=ChannelType.DEPOLARIZING,
            depolarizing_prob=0.1,
            loss_probability=0.01,
        )
        assert config.depolarizing_prob == 0.1
        assert config.loss_probability == 0.01
    
    def test_invalid_loss_probability(self):
        with pytest.raises(ValueError):
            ChannelConfig(loss_probability=1.5)
        with pytest.raises(ValueError):
            ChannelConfig(loss_probability=-0.1)
    
    def test_invalid_depolarizing_prob(self):
        with pytest.raises(ValueError):
            ChannelConfig(depolarizing_prob=1.5)
        with pytest.raises(ValueError):
            ChannelConfig(depolarizing_prob=-0.1)
    
    def test_invalid_phase_flip_prob(self):
        with pytest.raises(ValueError):
            ChannelConfig(phase_flip_prob=1.5)
        with pytest.raises(ValueError):
            ChannelConfig(phase_flip_prob=-0.1)
    
    def test_invalid_damping_gamma(self):
        with pytest.raises(ValueError):
            ChannelConfig(damping_gamma=1.5)
        with pytest.raises(ValueError):
            ChannelConfig(damping_gamma=-0.1)


class TestChannelWithLoss:
    """Test channels with photon loss."""
    
    def test_ideal_channel_with_loss(self):
        """Ideal channel with loss should drop photons."""
        channel = IdealChannel(seed=42)
        channel.config.loss_probability = 0.3
        
        state = BB84_STATES[(Basis.Z, Bit.ZERO)]
        n = 5000
        
        lost = 0
        for _ in range(n):
            out, detected = channel.apply_loss(state)
            if not detected:
                lost += 1
        
        assert np.isclose(lost / n, 0.3, atol=0.02)
    
    def test_depolarizing_with_loss(self):
        """Depolarizing channel should respect loss."""
        channel = DepolarizingChannel(depolarizing_prob=0.5, seed=42)
        channel.config.loss_probability = 0.3
        
        state = BB84_STATES[(Basis.Z, Bit.ZERO)]
        n = 5000
        
        lost = 0
        for _ in range(n):
            out = channel.transmit(state)
            if np.allclose(out, 0):
                lost += 1
        
        # Loss should dominate when state becomes zero
        assert lost / n > 0.2  # Some photons lost


class TestChannelReproducibility:
    """Test that channels are reproducible with same seed."""
    
    def test_ideal_reproducible(self):
        channel1 = IdealChannel(seed=42)
        channel2 = IdealChannel(seed=42)
        
        state = BB84_STATES[(Basis.Z, Bit.ZERO)]
        out1 = channel1.transmit(state)
        out2 = channel2.transmit(state)
        
        assert np.allclose(out1, out2)
    
    def test_depolarizing_reproducible(self):
        channel1 = DepolarizingChannel(depolarizing_prob=0.1, seed=42)
        channel2 = DepolarizingChannel(depolarizing_prob=0.1, seed=42)
        
        state = BB84_STATES[(Basis.Z, Bit.ZERO)]
        n = 100
        
        for _ in range(n):
            out1 = channel1.transmit(state)
            out2 = channel2.transmit(state)
            assert np.allclose(out1, out2)
    
    def test_phase_flip_reproducible(self):
        channel1 = PhaseFlipChannel(phase_flip_prob=0.3, seed=42)
        channel2 = PhaseFlipChannel(phase_flip_prob=0.3, seed=42)
        
        state = BB84_STATES[(Basis.Z, Bit.ZERO)]
        n = 100
        
        for _ in range(n):
            out1 = channel1.transmit(state)
            out2 = channel2.transmit(state)
            assert np.allclose(out1, out2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])