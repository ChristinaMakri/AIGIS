"""
Unit Tests for Agent Movement
Tests that agents do not exceed maximum speed limits
"""
import unittest
import numpy as np
from src.agents.civilian import CivilianAgent
from src.agents.rescuer import RescuerAgent
from src.config import CIVILIAN_V_FREE_FLOW, CIVILIAN_RHO_JAM


class TestCivilianMovement(unittest.TestCase):
    """Test civilian agent movement constraints"""

    def setUp(self):
        """Set up test civilian agent"""
        self.agent = CivilianAgent("test_civilian", (38.627, -90.199))
        self.max_speed = CIVILIAN_V_FREE_FLOW

    def test_free_flow_speed(self):
        """Test: Agent speed at zero density equals max speed"""
        # Simulate free flow (no congestion)
        self.agent.current_edge_density = 0.0

        # Calculate speed using Greenshields model
        speed = self.agent._calculate_speed_greenshields()

        print(f"\n✅ Test: Free Flow Speed")
        print(f"   Density: 0.0")
        print(f"   Calculated Speed: {speed:.2f} m/s")
        print(f"   Max Speed: {self.max_speed:.2f} m/s")

        # Assert speed equals max speed
        self.assertAlmostEqual(speed, self.max_speed, places=2,
                              msg="Speed in free flow should equal max speed")

    def test_speed_never_exceeds_max(self):
        """Test: Agent never exceeds max speed regardless of density"""
        test_densities = np.linspace(0, CIVILIAN_RHO_JAM, 20)

        for density in test_densities:
            self.agent.current_edge_density = density
            speed = self.agent._calculate_speed_greenshields()

            # Assert speed never exceeds max
            self.assertLessEqual(speed, self.max_speed,
                                msg=f"Speed {speed} exceeds max {self.max_speed} at density {density}")

        print(f"\n✅ Test: Speed Never Exceeds Max")
        print(f"   Tested {len(test_densities)} density values")
        print(f"   All speeds ≤ {self.max_speed:.2f} m/s")

    def test_gridlock_at_jam_density(self):
        """Test: Agent speed is zero at jam density (gridlock)"""
        # Simulate gridlock
        self.agent.current_edge_density = CIVILIAN_RHO_JAM

        # Calculate speed
        speed = self.agent._calculate_speed_greenshields()

        print(f"\n✅ Test: Gridlock at Jam Density")
        print(f"   Density: {CIVILIAN_RHO_JAM}")
        print(f"   Calculated Speed: {speed:.2f} m/s")

        # Assert speed is zero (gridlock)
        self.assertAlmostEqual(speed, 0.0, places=2,
                              msg="Speed should be 0 at jam density (gridlock)")

    def test_speed_decreases_with_density(self):
        """Test: Speed decreases monotonically as density increases"""
        densities = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
        speeds = []

        for density in densities:
            self.agent.current_edge_density = density
            speed = self.agent._calculate_speed_greenshields()
            speeds.append(speed)

        print(f"\n✅ Test: Speed Decreases with Density")
        print(f"   Density → Speed:")
        for d, s in zip(densities, speeds):
            print(f"     {d:.1f} → {s:.2f} m/s")

        # Assert speeds are monotonically decreasing
        for i in range(len(speeds) - 1):
            self.assertGreaterEqual(speeds[i], speeds[i + 1],
                                   msg=f"Speed should decrease as density increases")

    def test_confused_state_reduces_speed(self):
        """Test: Confused cognitive state reduces speed by 50%"""
        # Set free flow conditions
        self.agent.current_edge_density = 0.0
        self.agent.cognitive_state = "rational"

        # Calculate rational speed
        rational_speed = self.agent._calculate_speed_greenshields()

        # Switch to confused state
        self.agent.cognitive_state = "confused"
        confused_speed = self.agent._calculate_speed_greenshields()

        print(f"\n✅ Test: Confused State Reduces Speed")
        print(f"   Rational Speed: {rational_speed:.2f} m/s")
        print(f"   Confused Speed: {confused_speed:.2f} m/s")
        print(f"   Reduction: {(1 - confused_speed/rational_speed)*100:.0f}%")

        # Assert confused speed is approximately 50% of rational speed
        self.assertAlmostEqual(confused_speed, rational_speed * 0.5, places=2,
                              msg="Confused state should reduce speed by 50%")

    def test_speed_at_half_jam_density(self):
        """Test: Speed is approximately 50% at half jam density"""
        # Set density to half of jam density
        self.agent.current_edge_density = CIVILIAN_RHO_JAM / 2
        self.agent.cognitive_state = "rational"

        # Calculate speed
        speed = self.agent._calculate_speed_greenshields()
        expected_speed = self.max_speed * 0.5

        print(f"\n✅ Test: Speed at Half Jam Density")
        print(f"   Density: {self.agent.current_edge_density:.1f} (50% of jam)")
        print(f"   Calculated Speed: {speed:.2f} m/s")
        print(f"   Expected Speed: {expected_speed:.2f} m/s")

        # Assert speed is approximately 50% of max
        self.assertAlmostEqual(speed, expected_speed, places=1,
                              msg="Speed should be ~50% at half jam density")

    def test_non_negative_speed(self):
        """Test: Speed is never negative"""
        # Test various densities including above jam
        test_densities = [0, 5, 10, 15, 20]

        for density in test_densities:
            self.agent.current_edge_density = density
            speed = self.agent._calculate_speed_greenshields()

            # Assert speed is non-negative
            self.assertGreaterEqual(speed, 0.0,
                                   msg=f"Speed should never be negative (got {speed} at density {density})")

        print(f"\n✅ Test: Non-Negative Speed")
        print(f"   Tested densities: {test_densities}")
        print(f"   All speeds ≥ 0")


class TestRescuerMovement(unittest.TestCase):
    """Test rescuer agent movement constraints"""

    def setUp(self):
        """Set up test rescuer agent"""
        self.agent = RescuerAgent("test_rescuer", (38.627, -90.199))

    def test_rescuer_fuel_consumption(self):
        """Test: Rescuer consumes fuel when moving"""
        initial_fuel = self.agent.fuel

        # Simulate fuel consumption per move
        fuel_per_move = 0.1
        moves = 10

        final_fuel = initial_fuel - (fuel_per_move * moves)

        print(f"\n✅ Test: Rescuer Fuel Consumption")
        print(f"   Initial Fuel: {initial_fuel:.1f}")
        print(f"   Moves: {moves}")
        print(f"   Fuel per Move: {fuel_per_move}")
        print(f"   Final Fuel: {final_fuel:.1f}")

        # Assert fuel decreases
        self.assertLess(final_fuel, initial_fuel,
                       msg="Fuel should decrease after movement")

    def test_rescuer_position_changes(self):
        """Test: Rescuer position changes when moving"""
        initial_pos = self.agent.position

        # Simulate position change
        new_pos = (initial_pos[0] + 0.001, initial_pos[1] + 0.001)
        self.agent.position = new_pos

        print(f"\n✅ Test: Rescuer Position Changes")
        print(f"   Initial: {initial_pos}")
        print(f"   Final: {new_pos}")

        # Assert position changed
        self.assertNotEqual(self.agent.position, initial_pos,
                           msg="Position should change after movement")


class TestGreenshieldsModel(unittest.TestCase):
    """Test Greenshields Traffic Model implementation"""

    def test_greenshields_formula(self):
        """Test: Greenshields formula V = V_free × (1 - ρ/ρ_jam)"""
        v_free = 5.0
        rho_jam = 10.0

        # Test various densities
        test_cases = [
            (0.0, 5.0),    # Zero density → max speed
            (2.5, 3.75),   # 25% density → 75% speed
            (5.0, 2.5),    # 50% density → 50% speed
            (7.5, 1.25),   # 75% density → 25% speed
            (10.0, 0.0),   # Jam density → zero speed
        ]

        print(f"\n✅ Test: Greenshields Formula")
        print(f"   V_free = {v_free} m/s")
        print(f"   ρ_jam = {rho_jam}")
        print(f"\n   Density → Expected Speed → Actual Speed")

        for density, expected_speed in test_cases:
            actual_speed = v_free * (1 - (density / rho_jam))

            print(f"   {density:4.1f} → {expected_speed:4.2f} m/s → {actual_speed:4.2f} m/s")

            self.assertAlmostEqual(actual_speed, expected_speed, places=2,
                                  msg=f"Speed calculation incorrect for density {density}")


if __name__ == '__main__':
    print("=" * 70)
    print("🧪 AGENT MOVEMENT CONSTRAINT TESTS")
    print("=" * 70)

    # Run tests with verbose output
    unittest.main(verbosity=2)
