import seaborn as sns
import matplotlib.pyplot as plt     

def plot_cm(cm , FIG_PATH, classes):
    f = plt.figure(figsize=(25, 25))
    ax= f.add_subplot()
    sns.heatmap(cm, annot=True, ax = ax, fmt='d'); #annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(classes, rotation='vertical'); 
    ax.yaxis.set_ticklabels(classes, rotation='horizontal');
    plt.savefig(FIG_PATH)