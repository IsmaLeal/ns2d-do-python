eps = 1e-4
dt = 1e-3
sigma = 1e8

# after 1st timestep
mode_ev_order = sigma / eps + 10 + eps
mode_order = 1 + dt * mode_ev_order

coef_ev_order = sigma + 10 * eps + eps ** 2
coef_order = eps + dt * coef_ev_order

# GS
new_coef_order = coef_order * mode_order

print(f"modes: {mode_order:.3e}\ncoefs: {coef_order:.3e}\ncoefs_GS: {new_coef_order:.3e}")
