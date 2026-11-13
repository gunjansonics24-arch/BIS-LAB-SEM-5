
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Startup stages and available choices
stages = {
    "Idea Validation": [
        {"name": "Online Survey", "days": 3, "cost": 1000, "risk": 0.10},
        {"name": "Customer Interviews", "days": 7, "cost": 5000, "risk": 0.05},
        {"name": "Market Research", "days": 14, "cost": 12000, "risk": 0.02}
    ],

    "Company Registration": [
        {"name": "Sole Proprietorship", "days": 2, "cost": 2000, "risk": 0.25},
        {"name": "Private Limited", "days": 10, "cost": 15000, "risk": 0.05},
        {"name": "LLP", "days": 7, "cost": 8000, "risk": 0.10}
    ],

    "Product Development": [
        {"name": "No-Code MVP", "days": 7, "cost": 5000, "risk": 0.20},
        {"name": "Freelancers", "days": 21, "cost": 30000, "risk": 0.10},
        {"name": "In-House Development", "days": 30, "cost": 15000, "risk": 0.08}
    ],

    "Funding": [
        {"name": "Bootstrapping", "days": 0, "cost": 0, "risk": 0.25},
        {"name": "Angel Investors", "days": 30, "cost": 5000, "risk": 0.12},
        {"name": "Business Loan", "days": 15, "cost": 10000, "risk": 0.20}
    ],

    "Marketing": [
        {"name": "Organic Marketing", "days": 14, "cost": 3000, "risk": 0.10},
        {"name": "Paid Advertisements", "days": 5, "cost": 20000, "risk": 0.15},
        {"name": "Influencer Marketing", "days": 10, "cost": 12000, "risk": 0.12}
    ]
}

stage_names = list(stages.keys())

# PSO parameters
NUM_PARTICLES = 30
ITERATIONS = 100

w = 0.7
c1 = 1.5
c2 = 1.5

dimensions = len(stage_names)

# Position limits
lower = np.zeros(dimensions)
upper = np.array([
    len(stages[name]) - 1 for name in stage_names
])

# Calculate fitness
def fitness(position):

    choices = np.rint(position).astype(int)

    total_days = 0
    total_cost = 0
    total_risk = 0

    for i, stage in enumerate(stage_names):

        option = stages[stage][choices[i]]

        total_days += option["days"]
        total_cost += option["cost"]
        total_risk += option["risk"]

    average_risk = total_risk / dimensions

    # Weighted objective function
    score = (
        0.5 * (total_days / 100)
        + 0.3 * (total_cost / 100000)
        + 0.2 * average_risk
    )

    return score


# Initialize particles
positions = np.random.uniform(
    lower, upper, (NUM_PARTICLES, dimensions)
)

velocities = np.random.uniform(
    -1, 1, (NUM_PARTICLES, dimensions)
)

# Personal best
pbest_positions = positions.copy()

pbest_scores = np.array([
    fitness(p) for p in positions
])

# Global best
best_index = np.argmin(pbest_scores)

gbest_position = pbest_positions[best_index].copy()
gbest_score = pbest_scores[best_index]

convergence = []

# PSO algorithm
for iteration in range(ITERATIONS):

    for i in range(NUM_PARTICLES):

        current_score = fitness(positions[i])

        # Update personal best
        if current_score < pbest_scores[i]:

            pbest_scores[i] = current_score
            pbest_positions[i] = positions[i].copy()

        # Update global best
        if current_score < gbest_score:

            gbest_score = current_score
            gbest_position = positions[i].copy()

        # Random numbers
        r1 = np.random.rand(dimensions)
        r2 = np.random.rand(dimensions)

        # Update velocity
        velocities[i] = (
            w * velocities[i]
            + c1 * r1 * (pbest_positions[i] - positions[i])
            + c2 * r2 * (gbest_position - positions[i])
        )

        # Update position
        positions[i] += velocities[i]

        positions[i] = np.clip(
            positions[i], lower, upper
        )

    convergence.append(gbest_score)

    print(
        "Iteration:", iteration + 1,
        "Best Fitness:", round(gbest_score, 4)
    )


# Display best startup plan
best_choices = np.rint(
    gbest_position
).astype(int)

total_days = 0
total_cost = 0
total_risk = 0

print("\nBEST STARTUP PLAN USING PSO")
print("--------------------------------")

for i, stage in enumerate(stage_names):

    option = stages[stage][best_choices[i]]

    print("\n" + stage)
    print("Selected Option:", option["name"])
    print("Time:", option["days"], "days")
    print("Cost: Rs.", option["cost"])
    print("Risk:", option["risk"] * 100, "%")

    total_days += option["days"]
    total_cost += option["cost"]
    total_risk += option["risk"]

print("\n--------------------------------")
print("TOTAL STARTUP REQUIREMENTS")
print("Total Time:", total_days, "days")
print("Total Cost: Rs.", total_cost)
print("Average Risk:", round(total_risk / dimensions * 100, 2), "%")
print("Best Fitness:", round(gbest_score, 4))


# Plot convergence
plt.figure(figsize=(9, 5))

plt.plot(convergence)

plt.xlabel("Iterations")
plt.ylabel("Best Fitness")
plt.title("PSO Optimization for Starting a Company")
plt.grid()
plt.show()
