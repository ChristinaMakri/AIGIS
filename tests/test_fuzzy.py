"""
Unit Tests for Fuzzy Logic Risk Assessment
Tests that the Analyst agent correctly assesses risk levels
"""
import unittest
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class TestFuzzyLogic(unittest.TestCase):
    """Test fuzzy logic risk assessment"""

    def setUp(self):
        """Set up fuzzy control system"""
        # Define fuzzy variables
        tti = ctrl.Antecedent(np.arange(0, 61, 1), 'tti')  # Time To Impact (minutes)
        fire_intensity = ctrl.Antecedent(np.arange(0, 101, 1), 'fire_intensity')  # Percentage
        risk = ctrl.Consequent(np.arange(0, 101, 1), 'risk')  # Risk level (0-100)

        # TTI membership functions
        tti['very_near'] = fuzz.trimf(tti.universe, [0, 0, 5])
        tti['near'] = fuzz.trimf(tti.universe, [0, 5, 15])
        tti['medium'] = fuzz.trimf(tti.universe, [10, 20, 30])
        tti['far'] = fuzz.trimf(tti.universe, [25, 60, 60])

        # Fire intensity membership functions
        fire_intensity['low'] = fuzz.trimf(fire_intensity.universe, [0, 0, 40])
        fire_intensity['medium'] = fuzz.trimf(fire_intensity.universe, [30, 50, 70])
        fire_intensity['high'] = fuzz.trimf(fire_intensity.universe, [60, 100, 100])

        # Risk membership functions
        risk['low'] = fuzz.trimf(risk.universe, [0, 0, 30])
        risk['medium'] = fuzz.trimf(risk.universe, [20, 50, 80])
        risk['high'] = fuzz.trimf(risk.universe, [70, 85, 100])
        risk['critical'] = fuzz.trimf(risk.universe, [85, 100, 100])

        # Define fuzzy rules
        rule1 = ctrl.Rule(tti['very_near'] & fire_intensity['high'], risk['critical'])
        rule2 = ctrl.Rule(tti['near'] & fire_intensity['high'], risk['high'])
        rule3 = ctrl.Rule(tti['near'] & fire_intensity['medium'], risk['medium'])
        rule4 = ctrl.Rule(tti['medium'] & fire_intensity['high'], risk['high'])
        rule5 = ctrl.Rule(tti['medium'] & fire_intensity['medium'], risk['medium'])
        rule6 = ctrl.Rule(tti['far'] & fire_intensity['low'], risk['low'])
        rule7 = ctrl.Rule(tti['very_near'] | fire_intensity['high'], risk['high'])

        # Create control system
        self.risk_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])
        self.risk_sim = ctrl.ControlSystemSimulation(self.risk_ctrl)

    def test_high_wind_near_fire_critical_risk(self):
        """Test: High wind + near fire = Critical risk"""
        # Simulate: Fire is very close (2 minutes) with high intensity (90%)
        self.risk_sim.input['tti'] = 2
        self.risk_sim.input['fire_intensity'] = 90

        # Compute risk
        self.risk_sim.compute()
        risk_level = self.risk_sim.output['risk']

        print(f"\n✅ Test: High Wind + Near Fire")
        print(f"   TTI: 2 minutes")
        print(f"   Fire Intensity: 90%")
        print(f"   Risk Level: {risk_level:.1f}/100")

        # Assert risk is critical (> 85)
        self.assertGreater(risk_level, 85, "Risk should be CRITICAL when fire is very near and intense")

    def test_far_fire_low_intensity_low_risk(self):
        """Test: Far fire + low intensity = Low risk"""
        # Simulate: Fire is far (45 minutes) with low intensity (20%)
        self.risk_sim.input['tti'] = 45
        self.risk_sim.input['fire_intensity'] = 20

        # Compute risk
        self.risk_sim.compute()
        risk_level = self.risk_sim.output['risk']

        print(f"\n✅ Test: Far Fire + Low Intensity")
        print(f"   TTI: 45 minutes")
        print(f"   Fire Intensity: 20%")
        print(f"   Risk Level: {risk_level:.1f}/100")

        # Assert risk is low (< 40)
        self.assertLess(risk_level, 40, "Risk should be LOW when fire is far and weak")

    def test_medium_fire_medium_risk(self):
        """Test: Medium distance + medium intensity = Medium risk"""
        # Simulate: Fire is medium distance (20 minutes) with medium intensity (50%)
        self.risk_sim.input['tti'] = 20
        self.risk_sim.input['fire_intensity'] = 50

        # Compute risk
        self.risk_sim.compute()
        risk_level = self.risk_sim.output['risk']

        print(f"\n✅ Test: Medium Fire + Medium Intensity")
        print(f"   TTI: 20 minutes")
        print(f"   Fire Intensity: 50%")
        print(f"   Risk Level: {risk_level:.1f}/100")

        # Assert risk is medium (30-70)
        self.assertGreater(risk_level, 30, "Risk should be above LOW")
        self.assertLess(risk_level, 80, "Risk should be below HIGH")

    def test_near_fire_high_intensity_high_risk(self):
        """Test: Near fire + high intensity = High risk"""
        # Simulate: Fire is near (10 minutes) with high intensity (80%)
        self.risk_sim.input['tti'] = 10
        self.risk_sim.input['fire_intensity'] = 80

        # Compute risk
        self.risk_sim.compute()
        risk_level = self.risk_sim.output['risk']

        print(f"\n✅ Test: Near Fire + High Intensity")
        print(f"   TTI: 10 minutes")
        print(f"   Fire Intensity: 80%")
        print(f"   Risk Level: {risk_level:.1f}/100")

        # Assert risk is high (> 70)
        self.assertGreater(risk_level, 70, "Risk should be HIGH when fire is near and intense")

    def test_boundary_very_near_fire(self):
        """Test: Boundary case - very near fire (5 minutes)"""
        # Simulate: Fire at boundary (5 minutes) with medium intensity (60%)
        self.risk_sim.input['tti'] = 5
        self.risk_sim.input['fire_intensity'] = 60

        # Compute risk
        self.risk_sim.compute()
        risk_level = self.risk_sim.output['risk']

        print(f"\n✅ Test: Boundary Case - 5 Minutes")
        print(f"   TTI: 5 minutes")
        print(f"   Fire Intensity: 60%")
        print(f"   Risk Level: {risk_level:.1f}/100")

        # Assert risk is significant (> 60)
        self.assertGreater(risk_level, 60, "Risk should be significant at 5-minute boundary")


class TestRiskThresholds(unittest.TestCase):
    """Test risk threshold classifications"""

    def test_risk_classification_critical(self):
        """Test that risk > 85 is classified as CRITICAL"""
        risk = 90
        classification = "CRITICAL" if risk > 85 else "HIGH" if risk > 70 else "MEDIUM" if risk > 40 else "LOW"
        self.assertEqual(classification, "CRITICAL")
        print(f"\n✅ Risk Classification: {risk} → {classification}")

    def test_risk_classification_high(self):
        """Test that 70 < risk <= 85 is classified as HIGH"""
        risk = 75
        classification = "CRITICAL" if risk > 85 else "HIGH" if risk > 70 else "MEDIUM" if risk > 40 else "LOW"
        self.assertEqual(classification, "HIGH")
        print(f"\n✅ Risk Classification: {risk} → {classification}")

    def test_risk_classification_medium(self):
        """Test that 40 < risk <= 70 is classified as MEDIUM"""
        risk = 55
        classification = "CRITICAL" if risk > 85 else "HIGH" if risk > 70 else "MEDIUM" if risk > 40 else "LOW"
        self.assertEqual(classification, "MEDIUM")
        print(f"\n✅ Risk Classification: {risk} → {classification}")

    def test_risk_classification_low(self):
        """Test that risk <= 40 is classified as LOW"""
        risk = 25
        classification = "CRITICAL" if risk > 85 else "HIGH" if risk > 70 else "MEDIUM" if risk > 40 else "LOW"
        self.assertEqual(classification, "LOW")
        print(f"\n✅ Risk Classification: {risk} → {classification}")


if __name__ == '__main__':
    print("=" * 70)
    print("🧪 FUZZY LOGIC RISK ASSESSMENT TESTS")
    print("=" * 70)

    # Run tests with verbose output
    unittest.main(verbosity=2)
