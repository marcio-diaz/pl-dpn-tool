import unittest
import glob
from pldpn import *

class TestRaceDetectionOnFile(unittest.TestCase):

    def setUp(self):
        populate_config()
        
    def test_small_race(self):
        state = State()
        process_file('./tests/small_race.c', state)
        pldpn = PLDPN(control_states=state.control_states,
                      gamma=state.gamma,
                      rules=state.rules,
                      spawn_end_gamma=state.spawn_end_gamma)
        result = run_race_detection(pldpn, state.global_vars)
        self.assertTrue(result)

    def test_small_no_race(self):
        state = State()
        process_file('./tests/small_no_race.c', state)
        pldpn = PLDPN(control_states=state.control_states,
                      gamma=state.gamma,
                      rules=state.rules,
                      spawn_end_gamma=state.spawn_end_gamma)
        result = run_race_detection(pldpn, state.global_vars)
        self.assertFalse(result)

        
if __name__ == "__main__":
    unittest.main()
