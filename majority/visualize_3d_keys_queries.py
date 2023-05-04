import json
import numpy as np
import matplotlib.pyplot as plt


def visualize_query_keys(queries, keys):
    plt.clf()
    xs, ys, zs = keys
    x_q, y_q, z_q = queries
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for i in range(len(xs)):
        ax.plot([0, xs[i]], [0, ys[i]], [0, zs[i]], linewidth=1, markersize=5, marker='.', color='red', label='keys')
    for i in range(len(x_q)):
        ax.plot([0, x_q[i]], [0, y_q[i]], [0, z_q[i]], linewidth=1, markersize=5, marker='2', color='black', label='queries')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim3d(-3, 3)
    ax.set_ylim3d(-3, 3)
    ax.set_zlim3d(-3, 3)

    return ax, fig

if __name__ == '__main__':
    file_name = 'keys_queries.jsonl'
    prefixes = ['WITH_PROJ', 'WITHOUT_PROJ']
    for prefix in prefixes:
        with open(f'{prefix}_{file_name}', "r") as f:
            data_list = json.load(f)
        out = f"{prefix}_keys_queries.png"
        keys = []
        queries = []
        for run in data_list:
            keys.append(np.array(run['keys'])[0])
            queries.append(np.array(run['queries'])[0])
        
        keys = np.concatenate(keys, axis=0)
        queries = np.concatenate(queries, axis=0)

        ax = visualize_query_keys(queries.T, keys.T)
        plt.savefig(out)