from rlscore.measure.sqerror_measure import *
from rlscore.test.test_measure.abstract_measure_test import AbstractMultiTaskMeasureTest

class Test(AbstractMultiTaskMeasureTest):
    
    def setUp(self):
        AbstractMultiTaskMeasureTest.setUp(self)
        self.func = sqerror
        self.func_singletask = sqerror_singletask
        self.func_multitask = sqerror_multitask