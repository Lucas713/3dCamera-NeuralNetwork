#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:24:21 2020

@author: nvidia
"""

import servos


servos.initialise_shield()
servos.initialise_servos()

step_servos = 5 #pas de prise de mesure
num_servos = [3, 7, 11, 15] #numéros des pins des servos
pos_servos = [0, 0, 0, 0] # position des servos

for compteur in range(0,step_servos,180):
    

















