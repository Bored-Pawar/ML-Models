import pyomo.environ as pyo
import pandas as pd
import matplotlib.pyplot as plt

# store results
final_results = []

# constants
slots_per_day = 48
toatal_slots = slots_per_day * 8
initial_soc = 50 #MWh
prev_energy_level = initial_soc

# load data
df = pd.read_csv(r"C:\Users\Aditya Pramod Pawar\Machine Learning\Operations_Research\short_term_rolling_model\battery_market_385rows.csv")

# loop to go over each slot ie model runs for each 30 min slot again
for current_slot in range(toatal_slots - slots_per_day):
    horizon_df = df.iloc[current_slot : current_slot + slots_per_day].copy()

    # boolean variable for final day
    is_last_window = (current_slot == toatal_slots - slots_per_day - 1)

    # build / initialize model
    model = pyo.ConcreteModel()

    # time index
    model.T =  pyo.RangeSet(0, slots_per_day - 1) # RangeSets in pyomo is used to define timestamps

    # defining parameters
    battery_capacity = 100 # MWh
    max_charge_rate = 20 # MW
    max_discharge_rate = 20 # MW
    eta_charge = 0.9 # 90% percent efficiency
    eta_discharge = 0.9 # 90% percent efficiency
    final_energy = 50 #MWh same energy we strated with
    time_step = 0.5 # 30 minuite slot

    # add these parameters to model
    model.battery_capacity = pyo.Param(initialize = battery_capacity)
    model.max_charge_rate = pyo.Param(initialize = max_charge_rate)
    model.max_discharge_rate = pyo.Param(initialize = max_discharge_rate)
    model.eta_charge = pyo.Param(initialize = eta_charge)
    model.eta_discharge = pyo.Param(initialize = eta_discharge)
    model.final_energy = pyo.Param(initialize = final_energy)
    model.time_step = pyo.Param(initialize = time_step)

    # decision variable indexed by time they hold energy quantity
    model.charge = pyo.Var(model.T, within = pyo.NonNegativeReals, bounds = (0, max_charge_rate))
    model.discharge = pyo.Var(model.T, within = pyo.NonNegativeReals, bounds = (0, max_discharge_rate))
    model.energy = pyo.Var(model.T, within = pyo.NonNegativeReals, bounds = (0, battery_capacity))

    # binary decision variable
    model.u_charge = pyo.Var(model.T, within = pyo.Binary)
    model.u_discharge = pyo.Var(model.T, within = pyo.Binary)
    model.u_r_up = pyo.Var(model.T, within = pyo.Binary)
    model.u_r_down = pyo.Var(model.T, within = pyo.Binary)

    # reserve variables for holding energy quantity
    model.r_up = pyo.Var(model.T, within = pyo.NonNegativeReals)
    model.r_down = pyo.Var(model.T, within = pyo.NonNegativeReals)

    # market variables
    model.dam_purchase  = pyo.Var(model.T, within = pyo.NonNegativeReals)
    model.rt_purchase = pyo.Var(model.T, within = pyo.Reals)

    # load parameter data from horizon_df >>> loading in dictionary form {t : data[t]} where t is from 0 to 47
    dam_price = {t : horizon_df["DAM_Price"].iloc[t] for t in range(slots_per_day)}
    r_up_prices = {t : horizon_df["RU_Price"].iloc[t] for t in range(slots_per_day)}
    r_down_prices = {t : horizon_df["RD_Price"].iloc[t] for t in range(slots_per_day)}
    r_up_demand = {t : horizon_df["RU_Demand"].iloc[t] for t in range(slots_per_day)}
    r_down_demand = {t : horizon_df["RD_Demand"].iloc[t] for t in range(slots_per_day)}

    # load one real time price 
    rt_price = horizon_df["RT_Price"].iloc[0]

    # load dam and reserve prices and demand
    model.dam_price = pyo.Param(model.T, initialize = dam_price)
    model.r_up_prices = pyo.Param(model.T, initialize = r_up_prices)
    model.r_down_prices = pyo.Param(model.T, initialize = r_down_prices)
    model.r_up_demand = pyo.Param(model.T, initialize = r_up_demand)
    model.r_down_demand = pyo.Param(model.T, initialize = r_down_demand)

    # load real time price in model
    model.rt_price = pyo.Param(initialize = rt_price)

    # # FIX THE INITIAL ENERGY FOR THIS WINDOW
    # model.energy[0].fix(prev_energy_level)

    # charge control
    def charge_control_rule(model, t):
        return model.charge[t] <= model.max_charge_rate * time_step * model.u_charge[t]
    model.charge_control_constraint = pyo.Constraint(model.T, rule = charge_control_rule)

    # discharge control
    def discharge_control_rule(model, t):
        return model.discharge[t] <= model.max_discharge_rate * time_step * model.u_discharge[t]
    model.discharge_control_constraint = pyo.Constraint(model.T, rule = discharge_control_rule)

    # define energy balance 
    def energy_babalnce_rule(model, t):
        if t == model.T.first():
            return model.energy[t] == prev_energy_level + (model.eta_charge * time_step * model.charge[t]) - (time_step * model.discharge[t] / model.eta_discharge)
        else:
            return model.energy[t] == model.energy[t -1] + (model.eta_charge * time_step * model.charge[t]) - (time_step * model.discharge[t] / model.eta_discharge)
    model.energy_balance_constraint = pyo.Constraint(model.T, rule = energy_babalnce_rule)

    # final energy constrain
    if is_last_window:
        def final_energy_rule(model):
            return model.energy[model.T.last()] == model.final_energy
        model.final_energy_constraint = pyo.Constraint(rule = final_energy_rule)

    # reserve up feasibility
    def r_up_rule(model, t):
        return model.energy[t] >= model.r_up_demand[t] * time_step * model.u_r_up[t] / model.eta_discharge
    model.r_up_constraint = pyo.Constraint(model.T, rule = r_up_rule)

    # reserved down feasibility
    def r_down_rule(model, t):
        return model.energy[t] <= model.battery_capacity - (model.r_down_demand[t] * time_step * model.u_r_down[t] * model.eta_charge)
    model.r_down_constraint = pyo.Constraint(model.T, rule = r_down_rule)

    # def reserve up commitment
    def r_up_commitment_rule(model, t):
        return model.r_up[t] <= model.r_up_demand[t] * model.u_r_up[t] * time_step
    model.r_up_commitment = pyo.Constraint(model.T, rule = r_up_commitment_rule)

    # def reserve down commitment 
    def r_down_commitment_rule(model, t):
        return model.r_down[t] <= model.r_down_demand[t] * model.u_r_down[t] * time_step
    model.r_down_commitment = pyo.Constraint(model.T, rule = r_down_commitment_rule)

    # opperation mutex
    def operation_mutex_rule(model, t):
        return model.u_r_up[t] + model.u_r_down[t] + model.u_charge[t] + model.u_discharge[t] <= 1
    model.reserve_mutex = pyo.Constraint(model.T, rule = operation_mutex_rule)

    # Objective Function
    def profit_rule(model):
        return sum(
            (model.dam_price[t] * model.discharge[t]) - (model.dam_price[t] * model.charge[t] / model.eta_charge) + (model.r_up[t] * model.r_up_prices[t]) + (model.r_down[t] * model.r_down_prices[t])
            for t in model.T
        )   
    model.objective = pyo.Objective(rule = profit_rule, sense = pyo.maximize)

    # solve
    solver = pyo.SolverFactory('highs')
    results = solver.solve(model, tee = False)

    # CHECK SOLVER STATUS (CRITICAL)
    if str(results.solver.termination_condition) != 'optimal':
        print(f"Solver failed at slot {current_slot}!")

    # After solving, check actions at t=0
    if current_slot == 0:  # Check first window
        print(f"Slot 0 - Charge: {pyo.value(model.charge[0])}, Discharge: {pyo.value(model.discharge[0])}")

    # Check energy progression
    print(f"Window {current_slot}: Energy[0]={pyo.value(model.energy[0]):.2f}, Energy[1]={pyo.value(model.energy[1]):.2f}")

    # extract final energy of this window
    # To this:
    next_energy = pyo.value(model.energy[1])  # SOC AFTER first action
    prev_energy_level = next_energy  # Use for next window

    # Prices at current slot
    r_up_price = r_up_prices[0]
    r_down_price = r_down_prices[0]

    # Power values from model
    charge = pyo.value(model.charge[0])  # in MW
    discharge = pyo.value(model.discharge[0])  # in MW
    r_up = pyo.value(model.r_up[0])
    r_down = pyo.value(model.r_down[0])

    # Energy revenue (settled at RT price)
    energy_revenue = (discharge * rt_price - charge * rt_price) * time_step  # MWh * $/MWh

    # Reserve revenue (retainer paid even if not activated)
    reserve_revenue = (r_up * r_up_price + r_down * r_down_price) * time_step

    # Total revenue
    total_revenue = energy_revenue + reserve_revenue
    
    slot_result = {
    'slot': current_slot,
    'charge (MW)': charge,
    'discharge (MW)': discharge,
    'energy (MWh)': pyo.value(model.energy[0]),
    'r_up (MW)': r_up,
    'r_down (MW)': r_down,
    'u_charge': pyo.value(model.u_charge[0]),
    'u_discharge': pyo.value(model.u_discharge[0]),
    'u_r_up': pyo.value(model.u_r_up[0]),
    'u_r_down': pyo.value(model.u_r_down[0]),
    'rt_price': rt_price,
    'r_up_price': r_up_price,
    'r_down_price': r_down_price,
    'energy_revenue': energy_revenue,
    'reserve_revenue': reserve_revenue,
    'total_revenue': total_revenue,
    'objective': pyo.value(model.objective) # NOTE: Objective is based on DAM prices (for scheduling), but energy revenue is based on RT prices (realized)
    }

    final_results.append(slot_result)

# Convert final_results to DataFrame
df_results = pd.DataFrame(final_results)

# Calculate reserve retainer earnings (capacity payments)
df_results['reserve_retainer'] = (df_results['r_up (MW)'] * df_results['r_up_price'] + 
                                  df_results['r_down (MW)'] * df_results['r_down_price']) * 0.5  # 0.5 for 30min

# Format all numeric columns
float_cols = df_results.select_dtypes(include='float').columns
df_results[float_cols] = df_results[float_cols].round(2)

def display_battery_results(df):
    print("\n" + " SAMPLE DATA (First 25 Slots) ".center(130, '─'))
    
    sample_cols = ['slot', 'charge (MW)', 'discharge (MW)', 'energy (MWh)',
                   'r_up (MW)', 'r_down (MW)', 'reserve_retainer', 'total_revenue']
    
    col_widths = [15, 15, 15, 15, 15, 15, 18, 18]
    
    header = "│ " + " │ ".join([f"{col:^{width}}" for col, width in zip(sample_cols, col_widths)]) + " │"
    separator = "├" + "┼".join(["─"*(width+2) for width in col_widths]) + "┤"
    footer = "└" + "┴".join(["─"*(width+2) for width in col_widths]) + "┘"
    
    print(header)
    print(separator)
    
    for _, row in df[sample_cols].head(25).iterrows():
        row_str = "│ "
        for col, width in zip(sample_cols, col_widths):
            if col == 'slot':
                row_str += f"{int(row[col]):^{width}} │ "
            elif isinstance(row[col], float):
                if col in ['reserve_retainer', 'total_revenue']:
                    row_str += f"${row[col]:>{width-1}.2f} │ "
                else:
                    row_str += f"{row[col]:>{width}.2f} │ "
            else:
                row_str += f"{row[col]:^{width}} │ "
        print(row_str)
    
    print(footer)

# Display results
display_battery_results(df_results)

# Save to CSV
df_results.to_csv('battery_rolling_results.csv', index=False)
print("\nResults saved to 'battery_rolling_results.csv'")

# Plot with dots only (no lines for reserve retainer or charge/discharge)
plt.figure(figsize=(14, 6))
plt.plot(df_results['slot'], df_results['charge (MW)'], 'o', label='Charge (MW)')         # Dots only
plt.plot(df_results['slot'], df_results['discharge (MW)'], 'x', label='Discharge (MW)')   # Crosses only
plt.plot(df_results['slot'], df_results['energy (MWh)'], label='Energy (MWh)', linewidth=2)  # Line
plt.plot(df_results['slot'], df_results['reserve_retainer'], '^', label='Reserve Retainer ($)')  # Dots only

plt.xlabel("Time Slot")
plt.ylabel("Value")
plt.title("Battery Operation Summary (Zoomed to 0–100)")
plt.ylim(0, 100)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()