import pyomo.environ as pyo
import pandas as pd

# initializing pyomo model
model = pyo.ConcreteModel()

# load csv
df = pd.read_csv(r"C:\Users\Aditya Pramod Pawar\Machine Learning\Operations_Research\Long_term_model\energy_prices_week.csv")
T = list(df["Hour"]) # hours from 1 to 168
price_dict = {t : p for t, p in zip(df['Hour'], df["Price($/MWh)"])}

# adding parameters from csv to model
model.T = pyo.Set(initialize = T)
model.price = pyo.Param(model.T, initialize = price_dict, within = pyo.NonNegativeReals)

# defining new parameters
battery_capacity = 100 # MWh
max_charge_rate = 20 # MW
max_discharge_rate = 20 # MW
eta_charge = 0.9 # 90% percent efficiency
eta_discharge = 0.9 # 90% percent efficiency
initial_energy = 50 #MWh
final_energy = 50 #MWh same energy we strated with

# add them as parameter to the model
model.battery_capacity = pyo.Param(initialize = battery_capacity)
model.max_charge_rate = pyo.Param(initialize = max_charge_rate)
model.max_discharge_rate = pyo.Param(initialize = max_discharge_rate)
model.eta_charge = pyo.Param(initialize = eta_charge)
model.eta_discharge = pyo.Param(initialize = eta_discharge)
model.initial_energy = pyo.Param(initialize = initial_energy)

# decision variable (indexed by time)
model.charge = pyo.Var(model.T, within = pyo.NonNegativeReals, bounds = (0, model.max_charge_rate))
model.discharge = pyo.Var(model.T, within = pyo.NonNegativeReals, bounds = (0, model.max_discharge_rate))
model.energy = pyo.Var(model.T, within = pyo.NonNegativeReals, bounds = (0, model.battery_capacity))

# model.energy isn't a python parameter so can't be directly so we need a function
def initial_energy_rule(m):
    return m.energy[1] == m.initial_energy # pyomo parameter can't be assigned they can only be related using ==

model.initial_energy_constraint = pyo.Constraint(rule = initial_energy_rule) # here rule = initial.... tells pyomo to apply logic from inital_energy...

# creating binary decision variable
model.u_charge = pyo.Var(model.T, within = pyo.Binary)
model.u_discharge = pyo.Var(model.T, within = pyo.Binary)

# # logic to control charge/discharge using binary variable
# model.charge[t] <= max_charge_rate * model.u_charge[t]
# model.discharge[t] <= max_discharge_rate * model.u_discharge[t]

# # to prevent both from happening simultaneously
# u_charge + u_discharge <= 1

# the above code didn't work becoz for logic in pyomo needs to be added by making function and adding them as constraints

# control charge using binary variable
def charge_control_rule(model, t):
    return model.charge[t] <= max_charge_rate * model.u_charge[t]
model.charge_control = pyo.Constraint(model.T, rule = charge_control_rule)

# control discharge using binary variable
def discharge_control_rule(model, t):
    return model.discharge[t] <= max_discharge_rate * model.u_discharge[t]
model.discharge_control = pyo.Constraint(model.T, rule = discharge_control_rule)

# prevent simultaneous charge and discharge
def mutually_exclusice_rule(model, t):
    return model.u_charge[t] + model.u_discharge[t] <= 1
model.charge_discharge_mutex = pyo.Constraint(model.T, rule = mutually_exclusice_rule)

# creating energy balance constraint
def energy_balance_rule(model, t):
    if t == model.T.first():
        return model.energy[t] == model.initial_energy + (model.charge[t] * model.eta_charge) - (model.discharge[t] / model.eta_discharge)
    else:
        return model.energy[t] == model.energy[t - 1] + (model.charge[t] * model.eta_charge) - (model.discharge[t] / model.eta_discharge)
model.energy_balance = pyo.Constraint(model.T, rule = energy_balance_rule)   

# creating parameter for the final energy we want
model.final_energy = pyo.Param(initialize = final_energy)
def final_energy_contraint_rule(model, t):
    return model.energy[model.T.last()] == model.final_energy
model.final_energy_constraint = pyo.Constraint(model.T, rule = final_energy_contraint_rule)

# define battery max_limit as 100 and min_limit as 0
def energy_bounds_rule(model, t):
    return (0, model.energy[t], model.battery_capacity) # Pyomo reads this line as 0 <= energy[t] <= 100
model.energy_bounds = pyo.Constraint(model.T, rule = energy_bounds_rule)

# objective funtion
def profit_rule(model):
    return sum(
        (model.price[t] * model.discharge[t] * model.eta_discharge) - (model.price[t] * model.charge[t] / model.eta_charge)
        for t in model.T
    )
model.objective = pyo.Objective(rule = profit_rule, sense = pyo.maximize)

# create solver
solver = pyo.SolverFactory('highs')

# solve
results = solver.solve(model, tee = True)

# display solver status
print(results.solver.status)
print(results.solver.termination_condition)

# Display results for each hour
print(f"\n{'Hour':<5} {'Price':<6} {'Charge':<8} {'Discharge':<10} {'Energy':<7}")
for t in model.T:
    print(f"{t:<5} {pyo.value(model.price[t]):<6.2f} "
          f"{pyo.value(model.charge[t]):<8.2f} "
          f"{pyo.value(model.discharge[t]):<10.2f} "
          f"{pyo.value(model.energy[t]):<7.2f}")
    
# make csv file of result
data = {
    'Hour': list(model.T),
    'Price': [pyo.value(model.price[t]) for t in model.T],
    'Charge': [pyo.value(model.charge[t]) for t in model.T],
    'Discharge': [pyo.value(model.discharge[t]) for t in model.T],
    'Energy': [pyo.value(model.energy[t]) for t in model.T]
}

df = pd.DataFrame(data)
df.to_csv('results.csv', index=False)