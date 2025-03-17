import os
import pickle
import matplotlib
import matplotlib.pyplot as plt
import pandas
import numpy as np
import re

# Update matplotlib font settings
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

OURMETHOD = "PRESTO"
METHOD_MAPPING = {
    'ours_pcd_guide1': 'Point-Cloud Conditioning',
    'ours_nocost_guide1': 'Training Without TrajOpt',
    'ours_guide': OURMETHOD ,
}

def load_opt_data():

    # Function to convert the loaded data into the desired format with a mask
    def convert_data_with_mask(data, domain_to_obj):
        converted = {key: {} for key in METHOD_MAPPING.keys()}  # Predefine keys for diff 1 and 0
        for key, metric_value in data.items():

            domain_value, diff_value = key.split(",")
            domain = domain_value.split("=")[1]
            obj = domain_to_obj[domain]
            if metric_value['method'] not in METHOD_MAPPING.keys():
                continue
            diff = metric_value['method']

            # Store the data under the correct diff and obj keys
            converted[diff][obj] = {
                opt_iter: {
                'time': (metric_value['dt'][opt_iter], 0),
                'success_rate': (metric_value['suc_rate'][opt_iter], 0), # metric_value['suc_rate_std'][opt_iter]),
                'collision_rate': (metric_value['col_rate'][opt_iter], 0), # metric_value['col_rate_std'][opt_iter]),  # Assuming zero values
                'penetration_depth': (metric_value['avg_pen_cost'][opt_iter], 0), # metric_value['avg_pen_cost_std'][opt_iter]),     # Assuming zero values
                'collision_distance': (metric_value['avg_pen_cost'][opt_iter], 0), # metric_value['avg_pen_cost_std'][opt_iter]),  # Assuming zero values
                } for opt_iter in range(0, len(metric_value['dt']))
            }

        return {name: converted[key] for key, name in METHOD_MAPPING.items()}

    # Mapping of domain values to object names
    domain_to_obj = {
        "0.0": "obj-0-0",
        "1.0": "obj-1-1",
        "2.0": "obj-2-2",
        "3.0": "obj-3-4",
        "5.0": "obj-5-6"
    }

    with open("data/eval/proc_presto_guide.pkl", "rb") as file:
        opt_data = pickle.load(file)

    # Convert the loaded data into the desired format with a mask
    data = convert_data_with_mask(opt_data, domain_to_obj)
    return data


# Function to gather all the data points
def gather_data(data, domain, metric):
    times, values, std_devs = [], [], []
    if domain in data:
        for comp_time, metrics in data[domain].items():
            time_data = metrics['time'][0]  # Use the first value of the 'time' tuple as x-axis
            times.append(time_data)
            metric_data = metrics[metric]
            values.append(metric_data[0])  # Use the first value (average)
            std_devs.append(metric_data[1])  # Use the second value (std deviation)
    return times, values, std_devs


def plot_itr_data(data, metrics, iterations, domains):

    domain_names = {
        "obj-0-0": "Level 1",
        "obj-1-1": "Level 2",
        "obj-2-2": "Level 3",
        "obj-3-4": "Level 4",
        "obj-5-6": "Level 5"
    }

    # Make subplots for each domain (full range with inset)
    num_domains = len(domains)
    num_metrics = len(metrics)
    fig, axs = plt.subplots(num_metrics, num_domains*2, figsize=(3*1.5*num_domains, 2.5+1.3*(num_metrics-1)), gridspec_kw={'height_ratios': [1, 1, 1], 'width_ratios': [1.0]*num_domains+[0.25]*num_domains, 'hspace':0.3, 'wspace':0.3})
    plt.subplots_adjust(top=0.86, bottom=0.1, left=0.06, right=0.975)
    colors = {
        'Point-Cloud Conditioning': '#C59FE3',
        'Training Without TrajOpt': '#8C9AC6',
        OURMETHOD: 'deepskyblue',
    }

    handles, labels = [], []

    bar_width = 0.8 # Width of the bars
    spacing_line_bar = 0.04


    for j, metric in enumerate(metrics):
        for i, domain in enumerate(domains):
            # Main plot for full range (0%-100%)
            ax_line = axs[j, i]
            ax_bar = axs[j, i+num_domains]
            baseline_init_means = []

            for baseline_key, baseline_val in data.items():
                # Gather data for each baseline
                times, values_mean, values_std = gather_data(baseline_val, domain, metric)
                if times:
                    baseline_init_means.append(values_mean[0])
                    ax_line.plot(iterations[1:], values_mean[1:], label=baseline_key, color=colors[baseline_key], marker='o', markersize=4)

                    if baseline_key not in labels:
                        handles.append(ax_line.plot([], [], label=baseline_key, color=colors[baseline_key], marker='o')[0])
                        labels.append(baseline_key)

                    if metric == "success_rate" and baseline_key == OURMETHOD:
                        succ_ours = np.mean(values_mean[1:])
                    if metric == "success_rate" and baseline_key == "Point-Cloud Conditioning":
                        succ_point_cloud = np.mean(values_mean[1:])

            if metric == "success_rate":
                print(domain, np.array(succ_ours) - np.array(succ_point_cloud))
            ax_line.set_xticks(iterations[1:])
            # Show grid
            ax_line.grid(True)


            # Bar chart for iteration 0
            bar_positions = np.arange(len(baseline_init_means))
            bars = ax_bar.bar(bar_positions, baseline_init_means, width=bar_width, color=[colors[baseline_key] for baseline_key in data])

            # Remove x-axis labels from the bar chart
            ax_bar.set_xticks([])
            ax_bar.set_xlim(-1.0, 1.0 + (len(baseline_init_means)-1))
            ax_bar.grid(True)

            if i == 0:
                ax_line.set_ylabel(metric.replace('_', ' ').lower().capitalize(), size=12)
                ax_bar.set_ylabel(metric.replace('_', ' ').lower().capitalize(), size=12)
            if j == 0:
                ax_line.set_title(f"{domain_names[domain]}", style='italic', size=13)
                ax_bar.set_title(f"{domain_names[domain]}", style='italic', size=13)

            if metric == 'success_rate':
                ax_line.set_ylim(0.6, 1.0)
                ax_bar.set_ylim(0.0, 1.0)
            if metric == 'collision_rate':
                ax_line.set_ylim(0.0, 0.06)
                ax_bar.set_ylim(0.0, 0.3)
            if metric == 'penetration_depth':
                ax_line.set_ylim(0.0, 0.06)
                ax_bar.set_ylim(0.0, 0.15)

        # Adjust position of bar chart to create more space between the line and bar charts
        for i in range(0, num_domains):
            pos_line = axs[j, i].get_position()  # Get position of first bar chart
            axs[j, i].set_position([pos_line.x0 - 0.5*spacing_line_bar, pos_line.y0, pos_line.width, pos_line.height])  # Shift bar chart to the right

        for i in range(num_domains, 2* num_domains):
            pos_bar = axs[j, i].get_position()  # Get position of first bar chart
            axs[j, i].set_position([pos_bar.x0 + 0.5*spacing_line_bar, pos_bar.y0, pos_bar.width, pos_bar.height])  # Shift bar chart to the right

    # Make italic labels with spacing
    def italic_label(label):
        # Split by sequences of whitespace or a single dash, preserving them in the result
        tokens = re.split(r'(\s+|-)', label)
        # Wrap tokens in italic formatting unless they are a dash or whitespace
        new_tokens = [
            token if token == '-' or token.isspace() or token == '' 
            else f'$\\it{{{token}}}$' 
            for token in tokens
        ]
        new_label = ''.join(new_tokens)
        return new_label
    
    legend_labels = [label  if label is OURMETHOD else italic_label(label) for label in labels]

    legend_labels = [label + ' (With Guided-Sampling)' for label in legend_labels]

    # Add legend from collected handles and labels
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(labels), prop={'size': 12})

    # Annotate xlabel: Center the xlabel over the line charts (axs[j, 0] to axs[j, num_domains-1])
    left_line_ax = axs[0, 0]  # First line chart axis
    right_line_ax = axs[0, num_domains-1]  # Last line chart axis

    left_bar_ax = axs[0, num_domains]  # First line chart axis
    right_bar_ax = axs[0, 2*num_domains-1]  # Last line chart axis

    # Manually position the xlabel for line charts
    fig.text(
        (left_line_ax.get_position().x0 + right_line_ax.get_position().x1) / 2,
        0.02,  # Y position for xlabel (relative to figure height)
        'Number of trajectory optimization iterations', 
        # 'Time [sec] (log scale)', 
        ha='center', va='center', size=13
    )

    # Manually position the xlabel for bar charts
    fig.text(
        (left_bar_ax.get_position().x0 + right_bar_ax.get_position().x1) / 2,
        0.02,  # Y position for xlabel (same y as the line chart xlabel)
        'Initial trajectory', 
        ha='center', va='center', size=13
    )    

    # Save SVG
    plt.savefig('ablation_guidance.svg', format='svg', dpi=1200)

domains = ["obj-0-0", "obj-1-1", "obj-2-2", "obj-3-4"]  # Replace with actual domain values

# Call the plot function for success rates
data = load_opt_data()
plot_itr_data(data, metrics=['success_rate', 'collision_rate', 'penetration_depth'], domains=domains, iterations=[0, 1, 2, 3, 4, 5, 6, 7, 8])
