from rlscore.measure.accuracy_measure import *
from rlscore.test.test_measure.abstract_measure_test import AbstractMultiTaskMeasureTest

class Test(AbstractMultiTaskMeasureTest):
    
    def setUp(self):
        AbstractMultiTaskMeasureTest.setUp(self)
        self.func = accuracy
        self.func_singletask = accuracy_singletask
        self.func_multitask = accuracy_multitask

