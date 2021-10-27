import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt


core_temperature = 89
clock_speed = 3.8


graph_core_temp = np.arange(0, 101, 1)
ct_cold = fuzz.trimf(graph_core_temp, [0, 0, 50])
ct_warm = fuzz.trimf(graph_core_temp, [30, 50, 70])
ct_hot = fuzz.trimf(graph_core_temp, [50, 100, 100])

graph_clock_speed = np.arange(0, 5, 1)
cs_low = fuzz.trimf(graph_clock_speed, [0, 0, 1.5])
cs_warm = fuzz.trimf(graph_clock_speed, [0.5, 2, 3.5])
cs_hot = fuzz.trimf(graph_clock_speed, [2.5, 4, 4])

graph_fan_speed  = np.arange(0, 6001, 1)
fs_slow = fuzz.trimf(graph_fan_speed, [0, 0, 3500])
fs_fast = fuzz.trimf(graph_fan_speed, [2500, 6000, 6000])

# Visualize these universes and membership functions
fig, (graph0, graph1, graph2) = plt.subplots(nrows=3, figsize=(8, 9))

graph0.plot(graph_core_temp, ct_cold, 'b', linewidth=1.5, label='Cold')
graph0.plot(graph_core_temp, ct_warm, 'r', linewidth=1.5, label='Warm')
graph0.plot(graph_core_temp, ct_hot, 'y', linewidth=1.5, label='Hot')
graph0.set_title('Core Temperature')
graph0.legend()

graph1.plot(graph_clock_speed, cs_low, 'b', linewidth=1.5, label='Low')
graph1.plot(graph_clock_speed, cs_warm, 'r', linewidth=1.5, label='Warm')
graph1.plot(graph_clock_speed, cs_hot, 'y', linewidth=1.5, label='Hot')
graph1.set_title('Clock Speed')
graph1.legend()

graph2.plot(graph_fan_speed, fs_slow, 'b', linewidth=1.5, label='Slow')
graph2.plot(graph_fan_speed, fs_fast, 'r', linewidth=1.5, label='Fast')
graph2.set_title('Fan Speed')
graph2.legend()

for ax in (graph0, graph1, graph2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()


cold_temp = fuzz.interp_membership(graph_core_temp, ct_cold, core_temperature)
warm_temp = fuzz.interp_membership(graph_core_temp, ct_warm, core_temperature)
hot_temp = fuzz.interp_membership(graph_core_temp, ct_hot, core_temperature)

low_speed = fuzz.interp_membership(graph_clock_speed, cs_low, clock_speed)
warm_speed = fuzz.interp_membership(graph_clock_speed, cs_warm, clock_speed)
hot_speed = fuzz.interp_membership(graph_clock_speed, cs_hot, clock_speed)

active_rule1 = np.fmax(cold_temp, low_speed)

fan_speed_slow = np.fmin(active_rule1, fs_slow)

active_rule3 = np.fmax(hot_temp, hot_speed)
fan_speed_fast = np.fmin(active_rule3, fs_fast)
tip0 = np.zeros_like(graph_fan_speed)

fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.fill_between(graph_fan_speed, tip0, fan_speed_slow, facecolor='b', alpha=0.7)
ax0.plot(graph_fan_speed, fs_slow, 'b', linewidth=0.5, linestyle='--', )
ax0.fill_between(graph_fan_speed, tip0, fan_speed_fast, facecolor='r', alpha=0.7)
ax0.plot(graph_fan_speed, fs_fast, 'r', linewidth=0.5, linestyle='--')
ax0.set_title('Output membership activity')

for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()


# Aggregate all three output membership functions together
aggregated = np.fmax(fan_speed_slow, fan_speed_fast)

# Calculate defuzzified result
tip = fuzz.defuzz(graph_fan_speed, aggregated, 'centroid')
tip_activation = fuzz.interp_membership(graph_fan_speed, aggregated, tip)  # for plot

# Visualize this
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.plot(graph_fan_speed, fs_slow, 'b', linewidth=0.5, linestyle='--', )
ax0.plot(graph_fan_speed, fs_fast, 'r', linewidth=0.5, linestyle='--')
ax0.fill_between(graph_fan_speed, tip0, aggregated, facecolor='Orange', alpha=0.7)
ax0.plot([tip, tip], [0, tip_activation], 'k', linewidth=1.5, alpha=0.9)
ax0.set_title('Aggregated membership and result (line)')

# Turn off top/right axes
for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()


plt.show()