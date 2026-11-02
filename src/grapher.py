import matplotlib.pyplot as plt

linestyles = {'SGD': '-', 'SNGD': ':', 'DSNGD': '--', 'MAP': '-.', 'adaGrad': '-.', 'AdaGrad': '-.'}

color = {'SGD': 'C1', 'SNGD': 'C3', 'DSNGD': 'C2', 'MAP': 'C3', 'adaGrad': 'C5', 'AdaGrad': 'C5'}

# Known results of other models, drawn on the accuracy plot. Trained without distortions, the test error of LeNet-5
# (about 60,000 parameters) stabilizes at 0.95% after about 10 of its 20 passes through the training set
# (LeCun et al., Gradient-based learning applied to document recognition, Proc. IEEE 86(11), 1998).
REFERENCES = {'mnist': [{'label': 'LeNet-5 (LeCun et al., 1998)', 'accuracy': 1. - 0.0095, 'epochs': 10}]}


def plot_lines(x, graphs, labels, x_labels=False, y_labels=False, low_lines=None, high_lines=None, file_name=''):
    rows = len(graphs)
    columns = len(graphs[0])
    grid_num = rows*100 + columns*10 + 1
    # Each algorithm is labeled in the first subplot where it is drawn
    labeled = set()
    # plt.figure(figsize=(14, 3.9))
    for row in range(rows):
        for col in range(columns):
            ax = plt.subplot(grid_num + columns * row + col)
            # if col != 0:
            #     ax.get_yaxis().set_visible(False)
            # elif y_labels:
            if col == 0:
                plt.ylabel(y_labels[row], rotation=1)
            if row != rows-1:
                ax.get_xaxis().set_visible(False)
            elif x_labels:
                plt.xlabel(x_labels[col])

            lines = graphs[row][col]
            # convert y-axis to Logarithmic scale
            # plt.yscale("log")
            # convert y-axis to Logarithmic scale
            plt.xscale("log")
            # plt.xlim(10e3, 10e6)
            for l in range(len(lines)):
                line = lines[l]
                l_line = low_lines[row][col][l]
                h_line = high_lines[row][col][l]
                name = labels[l]
                if (line > 0).all():
                    plt.semilogy()
                    plt.plot(x, line, label=name if name not in labeled else "", color=color[name],
                             linestyle=linestyles[name])
                    labeled.add(name)
                    # uncomment below line to fill quartiles
                    plt.fill_between(x, l_line, h_line, facecolor=color[name], alpha=0.35)
    plt.figlegend(loc=8, ncol=len(labels))
    if file_name:
        try:
            f = file_name + '.png'
            plt.savefig(f)
            print("Graph saved: " + f)
        except Exception as e:
            print("ERROR : " + str(e))
            print("Couldn't save graph")
    else:
        print("No file given to save the graph. Showing the graph instead.")
        plt.show()
