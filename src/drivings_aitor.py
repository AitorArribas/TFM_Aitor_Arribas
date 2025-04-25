
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from braket.ahs.atom_arrangement import SiteType
from braket.ahs.driving_field import DrivingField
from braket.timings.time_series import TimeSeries

from braket.ahs.local_detuning import LocalDetuning
from braket.ahs.field import Field
from braket.ahs.pattern import Pattern
from collections import Counter

from braket.tasks.analog_hamiltonian_simulation_quantum_task_result import AnalogHamiltonianSimulationQuantumTaskResult
from braket.ahs.atom_arrangement import AtomArrangement



def Delta_local2(t, t_max, Delta_local_max, a, b):
    # assert 0 <= a <= t_max
    # assert a/2 <= b <= t_max - a/2

    if t >= 0 and t < b - a/2:
        return 0
    elif t >= b - a/2 and t < b + a/2:
        return Delta_local_max/2 + Delta_local_max/2*np.sin(np.pi/a*(t - b))
    elif t >= b + a/2:
        return Delta_local_max

                
def Delta2(t, t_max, Delta_max, a, b):
    if t >= 0 and t < b - a/2:
        return -Delta_max
    elif t >= b - a/2 and t < b + a/2:
        return -Delta_max/2 + Delta_max/2*np.sin(np.pi/a*(t - b))
    elif t >= b + a/2:
        return 0


def Drivings2(t, Omega_max, Delta_max, Delta_local_max, og_weights, Delta_0):
    t0 = 0
    time_max = t*1e-6
    
    # Parámetros del Delta global
    Delta_global_0 = (Delta_0[0] * t, Delta_0[1] * t)

    # Parámetros del Delta local
    Delta_local_0 = (Delta_0[2] * t, Delta_0[3] * t)

    
    
    time = np.linspace(t0, time_max, 100)
    ##MAXIMOS AQUILA
    #Omega_max = 15 *1e6  #RAD/S
    #Delta_max =  7.5 * 1e6 #RAD/S
    #Delta_local_max = 17 *1e6 #BEST
    
    omega_array = Omega_max*(np.sin((np.pi/2)*np.sin(np.pi*time/time_max)))**2
    delta_array = [Delta2(t, time_max, Delta_max, Delta_global_0[0]*1e-6, Delta_global_0[1]*1e-6) for t in time]
    delta_local_array = [Delta_local2(t, time_max, Delta_local_max, Delta_local_0[0]*1e-6, Delta_local_0[1]*1e-6) for t in time]
    #
    #-----------------------------
    phi_array = np.zeros_like(omega_array)
    
    
    omega = TimeSeries()
    
    for t_step, val in zip(time, omega_array):
        omega.put(t_step, val)
    
    
    global_detuning = TimeSeries()
    
    for t_step, val in zip(time, delta_array):
        global_detuning.put(t_step, val)
    
    phi = TimeSeries()

    for t_step, val in zip(time, phi_array):
        phi.put(t_step, val)
    

    # Asegurarse de que no hay elementos `None` en los valores de los drivings
    if any(v is None for v in delta_local_array):
        print(Delta_local_0)
        raise ValueError("`values` contiene elementos `None` no permitidos en delta_local_array.")
    if any(v is None for v in omega.values()):
        raise ValueError("`values` contiene elementos `None` no permitidos en omega.")
    if any(v is None for v in phi.values()):
        raise ValueError("`values` contiene elementos `None` no permitidos en phi.")
    if any(v is None for v in global_detuning.values()):
        raise ValueError("`values` contiene elementos `None` no permitidos en global_detuning.")
    
    
    # Defino los drivings global y local
    drive = DrivingField(amplitude=omega,
                         phase=phi,
                         detuning=global_detuning)
    
    local_detuning_drive = LocalDetuning.from_lists(times=time,
                                                        values=delta_local_array,
                                                        pattern=og_weights)
    

    return drive, local_detuning_drive, delta_local_array, time
