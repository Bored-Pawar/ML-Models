import pyomo.environ as pyo
import pandas as pd
import matplotlib.pyplot as plt

# initialize model
model = pyo.ConcreteModel()

# load csv
df = pd.read_csv(r"C:\Users\Aditya Pramod Pawar\Machine Learning\SMR_Operation_Research_model.py\smr_market_data_1_to_720.csv")
T = list(df["time"])
price_rt = {t : p for t, p in zip(df["time"], df["price_rt"])}
price_dam = {t : p for t, p in zip(df["time"], df["dam_price"])}
reserve_price = {t : p for t, p in zip(df["time"], df["reserve_price"])}
reserve_demand = {t : p for t, p in zip(df["time"], df["reserve_demand"])}

# intialize parameters from csv to model
model.T = pyo.Set(initialize = T)
model.price_rt = pyo.Param(model.T, initialize = price_rt, within = pyo.NonNegativeReals)
model.price_dam = pyo.Param(model.T, initialize = price_dam, within = pyo.NonNegativeReals)
model.reserve_price = pyo.Param(model.T, initialize = reserve_price, within = pyo.NonNegativeReals)
model.reserve_demand = pyo.Param(model.T, initialize = reserve_demand, within = pyo.NonNegativeReals)

# defining new SMR parameters
P_min = 20 # MW
P_max = 100 # MW
min_up_time = 6 # hours
min_down_time = 4
ramp_up_rate = 5 # MW/h
ramp_down_rate = 10 # MW/h
parasitic_load = 0.04 # energy lost
startup_energy = 6
shutdown_energy = 2
initial_power = 0

# define new economic parameter
operation_and_management_cost = 1800 # ruppes / MWh
startup_cost = 3000
shutdown_cost = 1000

# initialize all these variable in pyomo
model.P_min = pyo.Param(initialize = P_min)
model.P_max = pyo.Param(initialize = P_max)
model.min_up_time = pyo.Param(initialize = min_up_time)
model.min_down_time = pyo.Param(initialize = min_down_time)
model.ramp_up_rate = pyo.Param(initialize = ramp_up_rate)
model.ramp_down_rate = pyo.Param(initialize = ramp_down_rate)
model.parasitic_load = pyo.Param(initialize = parasitic_load)
model.startup_energy = pyo.Param(initialize = startup_energy)
model.shutdown_energy = pyo.Param(initialize = shutdown_energy)
model.initial_power = pyo.Param(initialize = initial_power)
model.operation_and_management_cost = pyo.Param(initialize = operation_and_management_cost)
model.startup_cost = pyo.Param(initialize = startup_cost)
model.shutdown_cost = pyo.Param(initialize = shutdown_cost)

# model decision variables
model.power = pyo.Var(model.T, within = pyo.NonNegativeReals, bounds = (0, model.P_max))
model.reserve = pyo.Var(model.T, within = pyo.NonNegativeReals, bounds = (0, model.P_max))
model.on = pyo.Var(model.T, within = pyo.Binary)
model.startup = pyo.Var(model.T, within = pyo.Binary)
model.shutdown = pyo.Var(model.T, within = pyo.Binary)

# contraints of the model 
# to initialize power at t = 1
def initial_power_rule(model):
    return model.power[1] == model.initial_power 
model.initial_power_constraint = pyo.Constraint(rule = initial_power_rule)

# p min constraint

# won't work because if statement comparing 
# def p_min_rule(model, t):
#     if t < 4:  # Need at least 4 time periods to check 4 consecutive startups
#         return pyo.Constraint.Skip
#     if sum(model.startup[t-i] for i in range(4)) >= 1:
#         return pyo.Constraint.Skip
#     else:
#         return (model.power[t] == 0) | (model.power[t] >= model.P_min)

def p_min_rule(model, t):
    if t < 4:  # Need at least 4 time periods to check 4 consecutive startups
        return pyo.Constraint.Skip
    startup_sum = sum(model.startup[t-i] for i in range(4))
    return model.power[t] >= model.P_min * (model.on[t] - startup_sum)
model.p_min_constraint = pyo.Constraint(model.T, rule = p_min_rule)

# # constraint to make sure that the prower doesn't just fluctuate in the 0 to 20 range in startup and reaches 20 on a compulsury baiss
# def ramp_up_efficiency_rule(model, t):
#     if t + 3 not in model.T:
#         return pyo.Constraint.Skip

#     expr = 0
#     for i in range(4):
#         if (t - i) in model.T:
#             expr += model.P_min * model.startup[t - i]
#     return model.power[t] >= expr
# model.ramp_up_efficiency_constraint = pyo.Constraint(model.T, rule = ramp_up_efficiency_rule)

# constraint for reactor on or off
def reactor_on_rule(model, t):
    return model.power[t] <= model.P_max * model.on[t]
model.reactor_on_constraint = pyo.Constraint(model.T, rule = reactor_on_rule)

# min up time


# in this code there is a issue because we are using if statements and the pyomo model can't work with python native if statement
# def min_up_time_rule(model, t):
#     expr = 0
#     for i in range(min_up_time):
#         if (t - i) in model.T:
#             expr += model.startup[t - i]
#     if expr > 0:
#         return model.on[t] == 1
#     else:
#         return pyo.Constraint.Skip
# model.min_up_time_constraint = pyo.Constraint(model.T, rule = min_up_time_rule)

# alternative corrected code 
def min_up_time_rule(model, t):
    if t < model.min_up_time:  # Skip if not enough history
        return pyo.Constraint.Skip
    
    # Sum startups in the last `min_up_time` periods
    startup_sum = sum(model.startup[t - i] for i in range(pyo.value(model.min_up_time))) # can't directly use model.min_up_time in range it is a pyo param we need to use pyo.value
    
    # If sum > 0, enforce `on[t] == 1` (using `startup_sum <= on[t]`)
    return startup_sum <= model.on[t]
model.min_up_time_constraint = pyo.Constraint(model.T, rule = min_up_time_rule)


# min down time


# in this code there is a issue because we are using if statements and the pyomo model can't work with python native if statement
# def min_down_time_rule(model, t):
#     expr = 0
#     for i in range(min_down_time):
#         if (t - i) in model.T:
#             expr += model.shutdown[t - i]
#     if expr > 0:
#         return model.on[t] == 0
#     else:
#         return pyo.Constraint.Skip
# model.min_down_time_constraint = pyo.Constraint(model.T, rule = min_down_time_rule)

# alternative corrected code 
def min_down_time_rule(model, t):
    if t < model.min_down_time:  # Skip if not enough history
        return pyo.Constraint.Skip
    
    # Sum startups in the last `min_up_time` periods
    shutdown_sum = sum(model.shutdown[t - i] for i in range(pyo.value(model.min_down_time))) # can't directly use model.min_down_time in range it is a pyo param we need to use pyo.value
    
    # If sum > 0, enforce `on[t] == 1` (using `startup_sum <= on[t]`)
    return shutdown_sum <= (1 - model.on[t])
model.min_down_time_constraint = pyo.Constraint(model.T, rule = min_down_time_rule)

# ramp up and down constraints
def ramp_up_rule(model, t):
    if t == model.T.first():
        return pyo.Constraint.Skip
    return model.power[t] - model.power[t-1] <= model.ramp_up_rate
model.ramp_up_constraint = pyo.Constraint(model.T, rule = ramp_up_rule)

def ramp_down_rule(model, t):
    if t == model.T.first():
        return pyo.Constraint.Skip
    return model.power[t-1] - model.power[t] <= model.ramp_down_rate + 10 * (1 - model.on[t])
model.ramp_down_constraint = pyo.Constraint(model.T, rule = ramp_down_rule)

# def startup or shutdown logic
def startup_shutdown_rule(model, t):
    if t == model.T.first():
        return pyo.Constraint.Skip
    return model.startup[t] - model.shutdown[t] == model.on[t] - model.on[t - 1]
model.startup_shutdown_constraint = pyo.Constraint(model.T, rule = startup_shutdown_rule)

# resrve should not exceed total generatable p
def reserve_limit_rule(model, t):
    return model.power[t] + model.reserve[t] <= model.P_max
model.reserve_limit_constraint = pyo.Constraint(model.T, rule = reserve_limit_rule)

# reserve offered should be less or equal to energy asked
def reserve_capacity_rule(model, t):
    return model.reserve[t] <= model.reserve_demand[t]
model.reserve_capacity_rule = pyo.Constraint(model.T, rule = reserve_capacity_rule)

# reserve offered should be less or equal to ramping up of energy possible
def reserve_ramp_up_rule(model, t):
    return model.reserve[t] <= model.ramp_up_rate * model.on[t]
model.reserve_ramp_up_constraint = pyo.Constraint(model.T, rule = reserve_ramp_up_rule)

# rule to keep startup and shutdown not 1 at the same time
def mutually_exclusive_rule(model, t):
    return model.startup[t] + model.shutdown[t] <= 1
model.mutually_exclusive_constraint = pyo.Constraint(model.T, rule = mutually_exclusive_rule)

def profit_rule(model, t):
    return sum(
        model.price_dam[t] * model.power[t] * (1 - model.parasitic_load) +
        model.reserve[t] * model.reserve_price[t] -
        model.power[t] * model.operation_and_management_cost -
        model.startup[t] * model.startup_energy * model.price_dam[t] -
        model.startup[t] * model.startup_cost -
        model.shutdown[t] * model.shutdown_energy * model.price_dam[t] -
        model.shutdown[t] * model.shutdown_cost
        for t in model.T
        )
model.profit = pyo.Objective(rule = profit_rule, sense = pyo.maximize)

# Create solver and solve
solver = pyo.SolverFactory('highs')
result = solver.solve(model, tee=True)

# Precompute constants
startup_cost = pyo.value(model.startup_cost)
shutdown_cost = pyo.value(model.shutdown_cost)
om_cost = pyo.value(model.operation_and_management_cost)

# Collect results
results = []

for t in model.T:
    on = pyo.value(model.on[t])
    power = pyo.value(model.power[t])
    reserve = pyo.value(model.reserve[t])
    startup = pyo.value(model.startup[t])
    shutdown = pyo.value(model.shutdown[t])
    rt_price = pyo.value(model.price_rt[t])
    reserve_price = pyo.value(model.reserve_price[t])

    revenue = (
        power * rt_price +
        reserve * reserve_price -
        startup * startup_cost -
        shutdown * shutdown_cost -
        on * om_cost -
        startup * startup_energy * rt_price -
        shutdown * shutdown_energy * rt_price
    )

    results.append({
        "Time": t,
        "On": int(on),
        "Power(MW)": round(power, 2),
        "Reserve(MW)": round(reserve, 2),
        "Startup": int(startup),
        "Shutdown": int(shutdown),
        "Revenue": round(revenue, 2)
    })

# Save to files
df = pd.DataFrame(results)
df.to_csv("smr_results.csv", index=False)
df.to_excel("smr_results.xlsx", index=False)

# Pretty print table
print("\n" + "_" * 89)
print("| {:^6} | {:^3} | {:^10} | {:^10} | {:^7} | {:^8} | {:^14} |".format(
    "Time", "On", "Power(MW)", "Reserve", "Start", "Shutdown", "Revenue"))
print("|" + "-"*6 + "|" + "-"*5 + "|" + "-"*12 + "|" + "-"*12 + "|" +
      "-"*9 + "|" + "-"*10 + "|" + "-"*16 + "|")

for row in results:
    print("| {:^6} | {:^3} | {:^10.2f} | {:^10.2f} | {:^7} | {:^8} | {:^14.2f} |".format(
        row["Time"], row["On"], row["Power(MW)"], row["Reserve(MW)"],
        row["Startup"], row["Shutdown"], row["Revenue"]))

print("|" + "_"*6 + "|" + "_"*5 + "|" + "_"*12 + "|" + "_"*12 + "|" +
      "_"*9 + "|" + "_"*10 + "|" + "_"*16 + "|")


# ------------------------------------------------------------MatPlotLib_Code----------------------------------------------------------------


# Load results again if needed
df = pd.read_csv("smr_results.csv")

# Create plot
fig, ax1 = plt.subplots(figsize=(15, 6))

# Plot power and reserve
ax1.plot(df["Time"], df["Power(MW)"], label="Power Output (MW)", color='tab:blue', linewidth=2)
ax1.plot(df["Time"], df["Reserve(MW)"], label="Reserve Offered (MW)", color='tab:orange', linewidth=2)

# Highlight when the reactor is off
for i in range(len(df)):
    if df["On"][i] == 0:
        ax1.axvspan(df["Time"][i] - 0.5, df["Time"][i] + 0.5, color='gray', alpha=0.1)

# Labels and formatting
ax1.set_xlabel("Time (Hours)", fontsize=12)
ax1.set_ylabel("MW", fontsize=12)
ax1.set_title("SMR Power and Reserve Offerings Over Time", fontsize=14)
ax1.legend()
ax1.grid(True)

# Optional: Secondary axis for ON/OFF status
ax2 = ax1.twinx()
ax2.plot(df["Time"], df["On"], label="Reactor ON", color='green', linestyle='--', alpha=0.5)
ax2.set_ylabel("Status (0=Off, 1=On)", fontsize=12)
ax2.set_yticks([0, 1])
ax2.set_ylim(-0.1, 1.1)

# Combine legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

plt.tight_layout()
plt.show()
