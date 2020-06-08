#!/usr/bin/env/python

import check_property

#real evaluation function
def evaluate_point(input,target=None):
    
    return check_property.check_property(input["smiles"], target=target)
