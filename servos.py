#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 14:48:38 2019

@author: nvidia
"""

import time
import smbus
from adafruit_servokit import ServoKit as sk
from time import sleep


#Configuration shield
shield=sk(channels=16, address=0x40)

#Configuration premier servo
#indique le domaine angulaire du servo sur les pins 3 (attention, ça commence au pin 0)
shield.servo[3].actuation_range=180
#indique l'intervalle de longueurs d'impulsions pour la PWM
shield.servo[3].set_pulse_width_range(500,2500)


#Configuration second servo
shield.servo[7].actuation_range=180
shield.servo[7].set_pulse_width_range(500,2500)


#Configuration troisième servo
shield.servo[11].actuation_range=180
shield.servo[11].set_pulse_width_range(500,2500)


#Configuration quatrième servo
shield.servo[15].actuation_range=180
shield.servo[15].set_pulse_width_range(500,2500)

def initialise_shield():
    #Configuration shield
    shield=sk(channels=16, address=0x40)
    #Configuration premier servo
    #indique le domaine angulaire du servo sur les pins 3 (attention, ça commence au pin 0)
    shield.servo[3].actuation_range=180
    #indique l'intervalle de longueurs d'impulsions pour la PWM
    shield.servo[3].set_pulse_width_range(500,2500)
    #Configuration second servo
    shield.servo[7].actuation_range=180
    shield.servo[7].set_pulse_width_range(500,2500)
    #Configuration troisième servo
    shield.servo[11].actuation_range=180
    shield.servo[11].set_pulse_width_range(500,2500)
    #Configuration quatrième servo
    shield.servo[15].actuation_range=180
    shield.servo[15].set_pulse_width_range(500,2500)

def angle(pin, angle):
    shield.servo[pin].angle=angle #cette fonction sert de raccourci à la commande en position

def initialise_servos(): #met tous les servos en position initiale (0°)
    angle(3,0)
    angle(7,0)
    angle(11,0)
    angle(15,0)
    
def uniforme(x):
    move(3,x)
    move(7,x)
    move(11,x)
    move(15,x)

def position(pin): #donne la position du servo au pin "pin"
    print("angle servo pin (°)", pin, ": ", shield.servo[pin].angle)

def course(pin):
    print("Position linéaire de l'actionneur ", pin, ": ", shield.servo[pin].angle/18, " cm")
    
def move(pin, course): #permet la commande en position linéaire des actionneurs (Attention : butées : 0 cm et 10 cm)
    #assert course>=0 & course<=10, "Entrer une course comprise entre 0 cm et 10 cm"
    angle(pin, 18*course)
    
def sequence():
    initialise_servos()
    uniforme(6)
    
    move(3,7.5)
    sleep(0.5)
    move(3,6)
    sleep(0.5)
    
    move(7,7.5)
    sleep(0.5)
    move(7,6)
    sleep(0.5)
    
    move(11,7.5)
    sleep(0.5)
    move(11,6)
    sleep(0.5)
    
    move(15,7.5)
    sleep(0.5)
    move(15,6)
    sleep(0.5)
    
    uniforme(7.5)
    sleep(0.5)
    uniforme(6)
    sleep(0.5)
    
    move(3,7.5)
    move(7,8)
    move(11,6)
    move(15,8)
    sleep(0.5)
    
    move(3,8)
    move(7,6)
    move(11,8)
    move(15,6)
    sleep(0.5)
    
    move(3,6)
    move(7,6)
    move(11,8)
    move(15,8)
    sleep(0.5)
    
    
    initialise_servos()

#move(3,8)
#move(7,5)
#move(11, 9)
#move(15,10)


#NOTE: 0° -> 0 cm de course et 180° -> 10 cm de course
# Donc course = (10/180)*angle servo
# soit x_i=(10/180)*q_i
