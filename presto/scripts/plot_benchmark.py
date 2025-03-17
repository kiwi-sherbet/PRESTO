import os
import pickle
import matplotlib
import matplotlib.pyplot as plt

PROC_FILE:str = 'data/eval/proc_presto.pkl'

# Update matplotlib font settings
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


OURMETHOD = "PRESTO"
METHOD_MAPPING = {
    'trajopt': 'TrajOpt',
    'sd': 'SceneDiffuser',
    'mpd_guide': 'MPD',
    'ours': OURMETHOD,
}


def load_rrt_data():

    def parse_log_file(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        result = {}
        time_stats = success_stats = collision_ratio_stats = collision_distance_stats = None
        error = None

        for line in lines:
            if "Statistics for time" in line:
                time_stats = tuple(float(x) for x in [
                    lines[lines.index(line) + 1].split(":")[1].strip(),
                    lines[lines.index(line) + 2].split(":")[1].strip(),
                    lines[lines.index(line) + 3].split(":")[1].strip(),
                    lines[lines.index(line) + 4].split(":")[1].strip()
                ])
            elif "Statistics for success" in line:
                success_stats = tuple(float(x) for x in [
                    lines[lines.index(line) + 1].split(":")[1].strip(),
                    lines[lines.index(line) + 2].split(":")[1].strip(),
                    lines[lines.index(line) + 3].split(":")[1].strip(),
                    lines[lines.index(line) + 4].split(":")[1].strip()
                ])
            elif "Statistics for collision_ratio" in line:
                collision_ratio_stats = tuple(float(x) for x in [
                    lines[lines.index(line) + 1].split(":")[1].strip(),
                    lines[lines.index(line) + 2].split(":")[1].strip(),
                    lines[lines.index(line) + 3].split(":")[1].strip(),
                    lines[lines.index(line) + 4].split(":")[1].strip()
                ])
            elif "Statistics for collision_distance" in line:
                collision_distance_stats = tuple(float(x) for x in [
                    lines[lines.index(line) + 1].split(":")[1].strip(),
                    lines[lines.index(line) + 2].split(":")[1].strip(),
                    lines[lines.index(line) + 3].split(":")[1].strip(),
                    lines[lines.index(line) + 4].split(":")[1].strip()
                ])
            elif "Statistics for error" in line:
                error = lines[lines.index(line)].split(":")[1].strip()
        
        result = {
                'time': time_stats,
                'success_rate': success_stats,
                'collision_rate': collision_ratio_stats,
                'penetration_depth': (0.0, 0.0, 0.0, 0.0),  # Assuming penetration data is missing or zero
                'collision_distance': collision_distance_stats,
                'error': error
        }
        
        return result

    def parse_multiple_logs(computation_times, objects):
        all_results = {obj: {} for obj in objects}
        
        for computation_time in computation_times:
            for obj in objects:
                file_path = f"data/eval/bi-rrt/bi-rrt_{computation_time}/{obj}/statistics.txt"
                
                if os.path.exists(file_path):
                    log_data = parse_log_file(file_path)
                    all_results[obj].update({computation_time: log_data})
                else:
                    print(f"File not found: {file_path}")
        
        return all_results

    # Example of usage:
    computation_times = [1, 2, 4, 8, 16, 32, 64, 128, 500]  # Replace with actual computation times
    objects = ["obj-0-0", "obj-1-1", "obj-2-2", "obj-3-4"]  # Replace with actual objects

    bi_rrt = parse_multiple_logs(computation_times, objects)

    return bi_rrt


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

    with open(PROC_FILE, "rb") as file:
        opt_data = pickle.load(file)

    # Convert the loaded data into the desired format with a mask
    data = convert_data_with_mask(opt_data, domain_to_obj)
    return data


def plot_data(bi_rrt, opt_data, metrics, domains):

    domain_names ={
        "obj-0-0": "Level 1",
        "obj-1-1": "Level 2",
        "obj-2-2": "Level 3",
        "obj-3-4": "Level 4",
        "obj-5-6": "Level 5"
    }

    # Function to gather all the data points from bi_rrt
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

    # Make subplots for each domain
    num_domains = len(domains)
    num_metrics = len(metrics)

    fig, axs = plt.subplots(num_metrics, num_domains, figsize=(3*1.5*num_domains, 2.5+1.5*(num_metrics-1)), 
                            gridspec_kw={'height_ratios': [1.5, 1.2, 1.2], 
                                         'hspace':0.3, 'wspace':0.3})

    # plt.subplots_adjust(top=0.8, bottom=0.175, left=0.075, right=0.975)
    plt.subplots_adjust(top=0.95, bottom=0.2, left=0.04, right=0.975)

    baselines = {'Bi-RRT': bi_rrt}
    baselines.update(opt_data)
    colors = {
        'Bi-RRT': 'darkred',
        'TrajOpt': 'olive',
        'SceneDiffuser': 'orange', 
        'MPD': 'purple', 
        OURMETHOD: 'blue',
    }
    handles, labels = [], []

    data_dict ={domain: {baseline:{} for baseline in baselines.keys()} for domain in domain_names.values()}

    for j, metric in enumerate(metrics):

        for i, domain in enumerate(domains):
            ax = axs[j, i]
            # Gather and plot presto data

            for baseline_key, baseline_val in baselines.items():
                # Gather and plot bi_rrt data
                times, values_mean, values_std = gather_data(baseline_val, domain, metric)
                if times:
                    # Sort the data points by time
                    if baseline_key != 'Bi-RRT':
                        times, values_mean, values_std = zip(*sorted(zip(times, values_mean, values_std)))

                    if baseline_key in ['SceneDiffuser', 'MPD']:
                        ax.axhline(values_mean[0], color=colors[baseline_key], linestyle='--')                        
                        ax.plot(times[0], values_mean[0], marker='o', color=colors[baseline_key], markersize=6)
                    elif baseline_key in ['TrajOpt']:
                        ax.plot(times[1:], values_mean[1:], label=baseline_key, color=colors[baseline_key], marker='o', markersize=4)
                    else:
                        ax.plot(times, values_mean, label=baseline_key, color=colors[baseline_key], marker='o', markersize=4)

                    # Mark the first point with a different color and add text "diffusion only"
                    if baseline_key == OURMETHOD:
                        ax.plot(times[0], values_mean[0], marker='o', color='black', markersize=6)

                    if baseline_key not in labels:
                        if baseline_key == OURMETHOD:
                            handles.append(ax.plot([], [], label="{} (Without Post-Processing)".format(baseline_key), color='black', linestyle='', marker='o')[0])
                            labels.append("{} (Without Post-Processing)".format(baseline_key))

                        if baseline_key in ['SceneDiffuser', 'MPD', 'Motion Planning Diffusion (Guided-Sampling)']:
                            handles.append(ax.plot([], color=colors[baseline_key], linestyle='--', markersize=6, marker='o')[0])
                            labels.append(baseline_key)
                        else:
                            handles.append(ax.plot([], [], label=baseline_key, color=colors[baseline_key], markersize=4, marker='o')[0])
                            labels.append(baseline_key)                    

                    if baseline_key in ['SceneDiffuser', 'MPD']:
                        data_dict[domain_names[domain]][baseline_key].update({'compuation_time': list(times[:1]), metric: list(values_mean[:1])})
                    else:
                        data_dict[domain_names[domain]][baseline_key].update({'compuation_time': list(times), metric: list(values_mean)})
                        if baseline_key != "Bi-RRT":
                            data_dict[domain_names[domain]][baseline_key].update({'iteration': [i for i in range(len(times))]})
                        else:
                            data_dict[domain_names[domain]][baseline_key].update({'timeout':  [1, 2, 4, 8, 16, 32, 64, 128, 500]})


                ax.set_yscale('linear')

                # Show the following x-ticks
                ax.set_xscale('log', base=10)
                ax.set_xticks([0.25, 0.5, 1, 2, 4, 10, 20, 40, 100, 200])
                ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

                if i == 0:
                    ax.set_ylabel(metric.replace('_', ' ').lower().capitalize(), size=12)
                if j == 0:
                    ax.set_title(f"{domain_names[domain]}", style='italic', size=13)

                if metric == 'success_rate':
                    ax.set_ylim(0.0, 1.0)

                ax.set_xlim(0.15,10)

                # Show grid
                ax.grid(True)

    # Add labels
    # Annotate xlabel: Center the xlabel over the line charts (axs[j, 0] to axs[j, num_domains-1])
    left_ax = axs[0, 0]  # First line chart axis
    right_ax = axs[0, num_domains-1]  # Last line chart axis

    fig.text(
        (left_ax.get_position().x0 + right_ax.get_position().x1) / 2,
        0.12,  # Y position for xlabel (relative to figure height)
        'Time [sec] (log scale)', 
        ha='center', va='center', size=13,
    )

    # Add legend from collected handles and labels
    fig.legend(handles, labels, loc="lower center",
                bbox_to_anchor=(0.5, 0.0), ncol=len(labels), prop={'size': 13})

    # Save SVG
    plt.savefig("benchmark.svg", format='svg', dpi=1200)

    with open(file='data.pickle', mode='wb') as f:
        pickle.dump(data_dict, f)

domains = ["obj-0-0", "obj-1-1", "obj-2-2", "obj-3-4"]  # Replace with actual domain values

# Call the plot function for success rates
bi_rrt = load_rrt_data()
data = load_opt_data()
plot_data(bi_rrt, data, metrics=[ 'success_rate', 'collision_rate', 'penetration_depth'], domains=domains)
