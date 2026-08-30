"""Build CENIC_working_notebook.docx. Run: PYTHONPATH=<dir with python-docx> python tools/make_notebook.py"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from nb_common import NB
from nb_part0_a import part0, partA
from nb_part_b import partB
from nb_part_c_end import partC, partD, partE, appendixF, appendixG, appendixH, appendixI

nb = NB()
part0(nb); partA(nb); partB(nb); partC(nb); partD(nb); partE(nb); appendixF(nb); appendixG(nb); appendixH(nb); appendixI(nb)
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CENIC_working_notebook.docx")
nb.save(out); print("wrote", out)
