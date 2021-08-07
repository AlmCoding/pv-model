
interest = 0.05
monthly_savings = 3000
monthly_expenses = 2000
net_worth = 100000

net_worth_trajectory = []
years = 0
while net_worth*interest < monthly_expenses*12:
    net_worth = net_worth * (1 + interest)
    net_worth += monthly_savings * 12
    net_worth_trajectory.append(int(net_worth))
    years += 1

print("Years until financial freedom: %i" % years)
print("Trajectory: ", net_worth_trajectory)
