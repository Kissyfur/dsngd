import matplotlib.pyplot as plt

linestyles = {'SGD': '-', 'SNGD': ':', 'DSNGD': '-.', 'MAP': '-.', 'AdaGrad': '-'}

color = {'SGD': 'C1', 'SNGD': 'C4', 'DSNGD': 'C2', 'MAP': 'C3', 'AdaGrad': 'C5'}


def plot_lines(x, graphs, labels, x_labels=False, y_labels=False, low_lines=None, high_lines=None, file_name=''):
    rows = graphs.shape[0]
    columns = graphs.shape[1]
    grid_num = rows*100 + columns*10 + 1
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

            lines = graphs[row, col]
            for l in range(len(lines)):
                line = lines[l]
                l_line = low_lines[row, col][l]
                h_line = high_lines[row, col][l]
                name = labels[l]
                if (line > 0).all():
                    plt.semilogy()
                    plt.plot(x, line, label=name if col + row == 0 else "", color=color[name], linestyle=linestyles[name])
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
