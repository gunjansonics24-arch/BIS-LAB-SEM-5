
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Room and AC settings
TARGET_TEMP = 24
INITIAL_TEMP = 30
ITERATIONS = 50
WOLVES = 10
TIME = 180

# Simulate room temperature and AC operation
def simulate(Kp, Ki):

    temp = INITIAL_TEMP
    integral = 0

    temperatures = []
    cooling = []
    energy = 0

    for t in range(TIME):

        # Outside temperature changes with time
        if t < 60:
            outside = 35
        elif t < 120:
            outside = 32
        else:
            outside = 38

        # Temperature error
        error = temp - TARGET_TEMP

        # PI controller
        integral = np.clip(integral + error, -100, 100)
        power = np.clip(Kp * error + Ki * integral, 0, 1)

        # Room temperature model
        temp_change = (
            0.04 * (outside - temp)
            + 0.08
            - 0.65 * power
        )

        temp += temp_change

        temperatures.append(temp)
        cooling.append(power)
        energy += power

    temperatures = np.array(temperatures)

    # Fitness: temperature error + energy consumption
    fitness = (
        np.mean((temperatures - TARGET_TEMP) ** 2)
        + 0.02 * energy / TIME
    )

    return fitness, temperatures, cooling


# Grey Wolf Optimizer
def GWO():

    # Each wolf represents [Kp, Ki]
    lower = np.array([0.02, 0.0])
    upper = np.array([0.8, 0.01])

    wolves = np.random.uniform(
        lower, upper, (WOLVES, 2)
    )

    alpha = np.zeros(2)
    beta = np.zeros(2)
    delta = np.zeros(2)

    alpha_score = float("inf")
    beta_score = float("inf")
    delta_score = float("inf")

    for iteration in range(ITERATIONS):

        for i in range(WOLVES):

            fitness, _, _ = simulate(
                wolves[i][0], wolves[i][1]
            )

            if fitness < alpha_score:
                delta_score = beta_score
                delta = beta.copy()

                beta_score = alpha_score
                beta = alpha.copy()

                alpha_score = fitness
                alpha = wolves[i].copy()

            elif fitness < beta_score:
                delta_score = beta_score
                delta = beta.copy()

                beta_score = fitness
                beta = wolves[i].copy()

            elif fitness < delta_score:
                delta_score = fitness
                delta = wolves[i].copy()

        a = 2 - iteration * (2 / ITERATIONS)

        for i in range(WOLVES):

            updated_positions = []

            for leader in [alpha, beta, delta]:

                r1 = np.random.rand(2)
                r2 = np.random.rand(2)

                A = 2 * a * r1 - a
                C = 2 * r2

                D = np.abs(C * leader - wolves[i])

                X = leader - A * D

                updated_positions.append(X)

            wolves[i] = np.mean(
                updated_positions, axis=0
            )

            wolves[i] = np.clip(
                wolves[i], lower, upper
            )

        print(
            "Iteration:", iteration + 1,
            "Best Fitness:", round(alpha_score, 4)
        )

    return alpha


# Run GWO
best_parameters = GWO()

best_Kp = best_parameters[0]
best_Ki = best_parameters[1]

# Simulate using optimized parameters
fitness, temperatures, cooling = simulate(
    best_Kp, best_Ki
)

print("\nOptimized AC Parameters")
print("Kp:", round(best_Kp, 4))
print("Ki:", round(best_Ki, 4))
print("Final Room Temperature:", round(temperatures[-1], 2), "°C")
print("Fitness:", round(fitness, 4))


# Plot room temperature
plt.figure(figsize=(10, 5))

plt.plot(temperatures, label="Room Temperature")
plt.axhline(
    y=TARGET_TEMP,
    linestyle="--",
    label="Target Temperature (24°C)"
)

plt.xlabel("Time (minutes)")
plt.ylabel("Temperature (°C)")
plt.title("Smart AC Temperature Control using GWO")
plt.legend()
plt.grid()
plt.show()


# Plot AC cooling power
plt.figure(figsize=(10, 5))

plt.plot(cooling, label="AC Cooling Power")
plt.xlabel("Time (minutes)")
plt.ylabel("Cooling Power (0 to 1)")
plt.title("Optimized AC Cooling Operation")
plt.legend()
plt.grid()
plt.show()
