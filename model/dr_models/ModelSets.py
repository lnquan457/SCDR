#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from model.dr_models.CDRs.nx_cdr import NxCDRModel
from model.dr_models.CDRs.cdr import CDRModel


MODELS = {
    "NX_CDR": NxCDRModel,
    "CDR": CDRModel,
}



