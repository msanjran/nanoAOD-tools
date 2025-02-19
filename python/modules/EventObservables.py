import os
import sys
import math
import json
import ROOT
import random

from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module

class EventObservables(Module):
    def __init__(self,
        leptonObject=None,
        metObject=None,
        ljets=None,
        
        outputName=None
    ):
        self.outputName=outputName

    def beginJob(self):
        pass

    def endJob(self):
        pass

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.out = wrappedOutputTree
        if self.outputName is not None:
            self.out.branch(self.outputName, "I")

    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        pass

    def analyze(self, event):
        if self.outputName is not None:
            self.out.fillBranch(self.outputName, self.passFilters(event))
            return True
        else:
            return self.passFilters(event)
