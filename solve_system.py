from LotkaVolterra import Lotka_Volterra

tmax = 30
timestep = 0.0001

predator_growth = 0.6
predator_death = 1.3
predator_initial = 0.1
prey_initial = 2

prey_growth = 0.8
prey_death = 0.8



L = Lotka_Volterra(predator_growth, predator_death, prey_growth,prey_death,tmax,timestep)
L.set_initial_conditions(predator_initial, prey_initial)
L.integrate()
L.plot_vs_time(filename = 'LT_vs_time.png')
L.plot_predator_vs_prey(filename = 'LT_PvsP.png')
L.save_data()
