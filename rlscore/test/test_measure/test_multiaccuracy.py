from rlscore.measure.multi_accuracy_measure import *
from rlscore.test.test_measure.abstract_measure_test import AbstractMeasureTest

class Test(AbstractMeasureTest):
    
    def setUp(self):
        AbstractMeasureTest.setUp(self)
        self.func = ova_accuracy