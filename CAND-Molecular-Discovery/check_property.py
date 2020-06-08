#!/usr/bin/env/python
from __future__ import print_function, division
from rdkit import Chem
import sascorer
from rdkit.Chem import Crippen
from rdkit.Chem import QED
import numpy as np

def check_property(smiles, target="logp"):
    if target=="validity":
        return check_validity(smiles)
    if target=="logp":
        return check_logp(smiles)
    elif target=="qed":
        return check_qed(smiles)
    elif target=="sa":
        return check_sascorer(smiles)
    elif target=="joint":
        #5*QED-SA
        return 5.0*check_qed(smiles) - check_sascorer(smiles)
    elif target=="logppunishedbysa":
        #logP-SA
        return check_logp(smiles) - check_sascorer(smiles)
    else:
        print("stop to evaluate :target error [%s]!"%target)
        exit(1)

def check_qed(smiles):
    new_mol=Chem.MolFromSmiles(smiles)
    try:
        val = QED.qed(new_mol)
    except:
        print("An error occurred in calculating the qed of molecule %s"%smiles)
        exit(1)
    return val

def check_sascorer(smiles):
    new_mol=Chem.MolFromSmiles(smiles)
    try:
        val = sascorer.calculateScore(new_mol)
    except:
        print("An error occurred in calculating the sa score of molecule %s"%smiles)
        exit(1)
    return val

def check_logp(smiles):
    new_mol=Chem.MolFromSmiles(smiles)
    try:
        val = Crippen.MolLogP(new_mol)
    except:
        print("An error occurred in calculating the logp of molecule %s"%smiles)
        exit(1)
    return val

def check_validity(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return 1.0
    return 0.0
