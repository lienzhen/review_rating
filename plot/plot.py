#coding=utf-8
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
#rcParams['font.family'] = 'STZhongSong'
#rcParams['font.sans-serif'] = ['SimHei']
rcParams["pdf.fonttype"] = 42

def plot_correlation():
    x = [100, 200, 300, 400, 500]
    y_w2v = [0.658445468, 0.67526367, 0.678470091, 0.679007418, 0.670645248]
    y_w2v_s = [0.70441986, 0.710108587, 0.683150336, 0.682185809, 0.677745905]
    y_w2v_r = [0.689281566, 0.691059701, 0.682926091, 0.680422257, 0.671778525]
    y_w2v_sr = [0.709785632, 0.710699037, 0.696636374, 0.691681258, 0.686750905]
    x_coor = [50, 100, 200, 300, 400, 500, 550]
    ax = plt.gca()
    ax.axis([50, 550, 0.6, 0.8])
    ax.set_xticks(x_coor)
    ax.set_xticklabels(('', '100', '200', '300', '400', '500', ''))
    #ax.set_xticklabels((100, 200, 300, 400, 500))
    ax.set_yticks(np.linspace(0.6, 0.8, 5))
    #ax.set_yticklabels(('0', '0.2', '0.4', '0.6', '0.8', '1.0'))
    ax.set_yticklabels(('0.6', '0.65', '0.7', '0.75', '0.8'))
    linestyle = '--'
    linewidth = 1.5
    markersize = 10
    mec='#FFFFFF'
    l_w2v = plt.plot(x, y_w2v, color='#9baec8', linewidth=linewidth, linestyle=linestyle, marker='o', ms=markersize, label='Word2vec', mec=mec)
    l_w2v_s = plt.plot(x, y_w2v_s, color='#ffc952', linewidth=linewidth, linestyle=linestyle, marker='v', ms=markersize, label='Word2vec+SWE', mec=mec)
    l_w2v_r = plt.plot(x, y_w2v_r, color='#47b8e0', linewidth=linewidth, linestyle=linestyle, marker='s', ms=markersize, label='Word2vec+RWE', mec=mec)
    l_w2v_sr = plt.plot(x, y_w2v_sr, color='#ff7473', linewidth=linewidth, linestyle=linestyle, marker='D', ms=markersize, label='Word2vec+SRWE', mec=mec)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::1], labels[::1])
    plt.xlabel('Dimension')
    plt.ylabel('Correlation')
    plt.title('Pearson correlation')
    plt.grid(True, color='#BDBDBD')
    filename = 'correlation.pdf'
    plt.savefig(filename, format='pdf')

def plot_topic_prediction():
    x = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    y_w2v = [0.7333, 0.9143, 0.9167, 0.9796, 0.9615, 0.9439, 0.9453, 0.9081, 0.9604, 0.9531]
    y_w2v_s = [0.8533, 0.9729, 0.9048, 0.9592, 0.9231, 0.9241, 0.9544, 0.9666, 0.989, 0.9531]
    y_w2v_r = [0.7867, 0.9474, 0.9048, 0.9184, 0.9231, 0.896, 0.9362, 0.9526, 0.9758, 0.9567]
    y_w2v_sr = [0.8667, 0.971, 0.9048, 0.9184, 0.9615, 0.9307, 0.9519, 0.961, 0.9846, 0.9458]
    x_coor = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1050]
    ax = plt.gca()
    ax.axis([50, 1050, 70, 100])
    ax.set_xticks(x_coor)
    ax.set_xticklabels(('', 'interests', 'biology', 'fashion', 'language', 'geology', 'food', 'computer', 'boats', 'astronomy', 'chemistry', ''), rotation=25)
    #ax.set_xticklabels((100, 200, 300, 400, 500))
    ax.set_yticks(np.linspace(70, 100, 7))
    #ax.set_yticklabels(('0', '0.2', '0.4', '0.6', '0.8', '1.0'))
    ax.set_yticklabels(('70', '75', '80', '85', '90', '95', '100'))
    linestyle = '--'
    linewidth = 0.8
    markersize = 10
    mec='#FFFFFF'
    l_w2v = plt.plot(x, [each * 100 for each in y_w2v], color='cyan', linewidth=linewidth, linestyle=linestyle, marker='o', ms=markersize, label='Word2vec', mec=mec)
    l_w2v_s = plt.plot(x, [each * 100 for each in y_w2v_s], color='blue', linewidth=linewidth, linestyle=linestyle, marker='v', ms=markersize, label='Word2vec+SWE', mec=mec)
    l_w2v_r = plt.plot(x, [each * 100 for each in y_w2v_r], color='green', linewidth=linewidth, linestyle=linestyle, marker='s', ms=markersize, label='Word2vec+RWE', mec=mec)
    l_w2v_sr = plt.plot(x, [each * 100 for each in y_w2v_sr], color='red', linewidth=linewidth, linestyle=linestyle, marker='D', ms=markersize, label='Word2vec+SRWE', mec=mec)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::1], labels[::1], loc='best')
    plt.xlabel('Domain')
    plt.ylabel('Accuracy(%)')
    plt.title('Domain Prediction (Dimension: 200)')
    plt.grid(True, color='#BDBDBD')
    filename = 'domain_prediction.pdf'
    plt.savefig(filename, format='pdf')

def plot_domain_acc():
    n_groups = 4
    # w2v w2v+s w2v+r w2v+sr
    #precision = np.array([0.7209, 0.7445, 0.7488, 0.7425])
    #recall = np.array([0.6805, 0.7092, 0.7158, 0.7127])
    #fscore = np.array([0.6971, 0.7242, 0.7299, 0.7255])
    #micro_precision = np.array([0.7595, 0.7825, 0.7851, 0.7825])

    dimension = 200
    if dimension == 100:
        w2v = np.array([0.7209, 0.6805, 0.6971, 0.7595])
        w2v_s = np.array([0.7345, 0.6992, 0.7142, 0.7725])
        w2v_r = np.array([0.7488, 0.7158, 0.7299, 0.7851])
        w2v_sr = np.array([0.7425, 0.7127, 0.7255, 0.7825])
    else:
    # 200 D
        w2v = np.array([0.7414, 0.7084, 0.7226, 0.7799])
        w2v_s = np.array([0.7606, 0.732, 0.7447, 0.7963])
        w2v_r = np.array([0.7714, 0.7427, 0.7555, 0.8066])
        w2v_sr = np.array([0.7708, 0.7429, 0.7553, 0.8071])

    fig, ax = plt.subplots()
    # for legend outside
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 1.2, box.height])

    index = np.arange(n_groups) + 0.5
    bar_width = 0.4
    opacity = 0.9
    rects1 = plt.bar(index, w2v*100, bar_width/2, alpha=opacity, color='#9baec8', label='Word2vec')
    rects2 = plt.bar(index + bar_width/2, w2v_s*100, bar_width/2, alpha=opacity, color='#ffc952', label='Word2vec+SWE')
    rects3 = plt.bar(index + bar_width, w2v_r*100, bar_width/2, alpha=opacity, color='#47b8e0', label='Word2vec+RWE')
    rects4 = plt.bar(index + bar_width*1.5, w2v_sr*100, bar_width/2, alpha=opacity, color='#ff7473', label='Word2vec+SRWE')
    plt.xticks(index+bar_width, ('Precision', 'Recall', 'F-score', 'Micro precision'), rotation=15)
    ax.axis([0, 5, 0, 100])
    plt.ylabel('%')
    plt.title('Classification result of NYTimes (%d Dimension)' % dimension)
    #plt.legend(bbox_to_anchor=(1.005, 0.9), loc='center left', borderaxespad=0., prop={'size':8})
    plt.legend(bbox_to_anchor=(0.79, 0.92), loc='center left', borderaxespad=0., prop={'size':8})
    plt.grid(True, color='#BDBDBD')
    plt.tight_layout()
    filename = 'text_classification_cmp_%d.pdf' % dimension
    plt.savefig(filename, format='pdf')

def plot_topic_predict_detail():
    n_groups = 10
    # w2v w2v+s w2v+r w2v+sr
    #precision = np.array([0.7209, 0.7445, 0.7488, 0.7425])
    #recall = np.array([0.6805, 0.7092, 0.7158, 0.7127])
    #fscore = np.array([0.6971, 0.7242, 0.7299, 0.7255])
    #micro_precision = np.array([0.7595, 0.7825, 0.7851, 0.7825])
    dimension = 200
    w2v = np.array([0.7333, 0.9143, 0.9167, 0.9796, 0.9615, 0.9439, 0.9453, 0.9081, 0.9604, 0.9531])
    w2v_s = np.array([0.8533, 0.9729, 0.9048, 0.9592, 0.9231, 0.9241, 0.9544, 0.9666, 0.989, 0.9531])
    w2v_r = np.array([0.7867, 0.9474, 0.9048, 0.9184, 0.9231, 0.896, 0.9362, 0.9526, 0.9758, 0.9567])
    w2v_sr = np.array([0.8667, 0.971, 0.9048, 0.9184, 0.9615, 0.9307, 0.9519, 0.961, 0.9846, 0.9458])

    fig, ax = plt.subplots()
    # for legend outside
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 1.2, box.height])

    index = np.arange(n_groups) * 2 + 0.5
    bar_width = 0.4
    opacity = 0.9
    rects1 = plt.barh(index, w2v*100, bar_width, alpha=opacity, color='#9baec8', label='Word2vec')
    rects2 = plt.barh(index + bar_width, w2v_s*100, bar_width, alpha=opacity, color='#ffc952', label='Word2vec+SWE')
    rects3 = plt.barh(index + bar_width*2, w2v_r*100, bar_width, alpha=opacity, color='#47b8e0', label='Word2vec+RWE')
    rects4 = plt.barh(index + bar_width*3, w2v_sr*100, bar_width, alpha=opacity, color='#ff7473', label='Word2vec+SRWE')
    plt.yticks(index+bar_width*2, ('interests', 'biology', 'fashion', 'language', 'geology', 'food', 'computer', 'boats', 'astronomy', 'chemistry', ''), rotation=0)
    ax.axis([70, 105, 0, 21])
    ax.set_xticks(np.arange(70, 110, 5))
    ax.set_xticklabels(('70', '75', '80', '85', '90', '95', '100', ''))
    plt.xlabel('Accuracy (%)')
    plt.title('Domain prediction (Dimension: %d)' % dimension)
    #plt.legend(bbox_to_anchor=(1.005, 0.9), loc='center left', borderaxespad=0., prop={'size':8})
    plt.legend(bbox_to_anchor=(0.785, 0.08), loc='center left', borderaxespad=0., prop={'size':8})
    plt.grid(True, color='#BDBDBD')
    plt.tight_layout()
    filename = 'topic_predict_detail_cmp_%d.pdf' % dimension
    plt.savefig(filename, format='pdf')

def sigmoid(x):
    return [1 / (1 + math.exp(-item)) for item in x]
def plot_sigmoid():
    x = np.arange(-10.0, 10.0, 0.2)
    sig = sigmoid(x)
    plt.plot(x, sig)
    plt.grid(True, color='#BDBDBD')
    plt.title('Sigmoid')
    plt.xlabel('$ x $')
    plt.ylabel('sigmoid($ x $)')
    plt.savefig('sigmoid.pdf', format='pdf')




def plot_small_dataset():
    x = [100, 200, 300, 400, 500]
    #y_w2v = [0.658445468, 0.67526367, 0.678470091, 0.679007418, 0.670645248]
    #y_w2v_s = [0.70441986, 0.710108587, 0.683150336, 0.682185809, 0.677745905]
    #y_w2v_r = [0.689281566, 0.691059701, 0.682926091, 0.680422257, 0.671778525]
    #y_w2v_sr = [0.709785632, 0.710699037, 0.696636374, 0.691681258, 0.686750905]
    models = []
    ys = []
    filename = 'sml_cmp1'
    with open('./%s' % filename) as fin:
        for line in fin:
            arr = line.strip().split('\t')
            models.append(arr[0])
            ys.append(map(lambda x: float(x), arr[1:]))
            #print ys[-1]
    x_coor = [50, 100, 200, 300, 400, 500, 550]
    ax = plt.gca()
    ax.axis([50, 550, 1.28, 1.48])
    ax.set_xticks(x_coor)
    ax.set_xticklabels(('', '100', '200', '300', '400', '500', ''))
    #ax.set_xticklabels((100, 200, 300, 400, 500))
    #ax.set_yticks(np.linspace(1.28, 1.41, 5))
    #ax.set_yticklabels(('0', '0.2', '0.4', '0.6', '0.8', '1.0'))
    #ax.set_yticklabels(('0.6', '0.65', '0.7', '0.75', '0.8'))
    linestyle = 'solid'
    linewidth = 1.5
    markersize = 10
    mec='#FFFFFF'
    markers = ['.', 'o', 'v', 's', 'D',]
    colors = ['#a3c9c7', '#FFBC42', '#D81159', '#8F2D56', '#218380']
    for i in range(5):
        plt.plot(x, ys[i], colors[i], linewidth=linewidth, linestyle='--' if i==0 else linestyle, marker=markers[i], ms=markersize, label=models[i], mec=mec)
    #l_w2v = plt.plot(x, y_w2v, color='#9baec8', linewidth=linewidth, linestyle=linestyle, marker='o', ms=markersize, label='Word2vec', mec=mec)
    #l_w2v_s = plt.plot(x, y_w2v_s, color='#ffc952', linewidth=linewidth, linestyle=linestyle, marker='v', ms=markersize, label='Word2vec+SWE', mec=mec)
    #l_w2v_r = plt.plot(x, y_w2v_r, color='#47b8e0', linewidth=linewidth, linestyle=linestyle, marker='s', ms=markersize, label='Word2vec+RWE', mec=mec)
    #l_w2v_sr = plt.plot(x, y_w2v_sr, color='#ff7473', linewidth=linewidth, linestyle=linestyle, marker='D', ms=markersize, label='Word2vec+SRWE', mec=mec)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::1], labels[::1])
    plt.xlabel('Dimension')
    plt.ylabel('MSE')
    plt.title('10k-Dataset Experiment Result')
    plt.grid(True, color='#BDBDBD')
    filename = '%s.png' % filename
    plt.savefig(filename, format='png')

def plot_medium_dataset():
    x = [100, 200, 300, 400, 500]
    #y_w2v = [0.658445468, 0.67526367, 0.678470091, 0.679007418, 0.670645248]
    #y_w2v_s = [0.70441986, 0.710108587, 0.683150336, 0.682185809, 0.677745905]
    #y_w2v_r = [0.689281566, 0.691059701, 0.682926091, 0.680422257, 0.671778525]
    #y_w2v_sr = [0.709785632, 0.710699037, 0.696636374, 0.691681258, 0.686750905]
    models = []
    ys = []
    filename = 'med_cmp1'
    with open('./%s' % filename) as fin:
        for line in fin:
            arr = line.strip().split('\t')
            models.append(arr[0])
            ys.append(map(lambda x: float(x), arr[1:]))
            #print ys[-1]
    x_coor = [50, 100, 200, 300, 400, 500, 550]
    ax = plt.gca()
    ax.axis([50, 550, 1.10, 1.40])
    ax.set_xticks(x_coor)
    ax.set_xticklabels(('', '100', '200', '300', '400', '500', ''))
    #ax.set_xticklabels((100, 200, 300, 400, 500))
    #ax.set_yticks(np.linspace(1.28, 1.41, 5))
    #ax.set_yticklabels(('0', '0.2', '0.4', '0.6', '0.8', '1.0'))
    #ax.set_yticklabels(('0.6', '0.65', '0.7', '0.75', '0.8'))
    linestyle = 'solid'
    linewidth = 1.5
    markersize = 10
    mec='#FFFFFF'
    markers = ['.', 'o', 'v', 's', 'D',]
    colors = ['#a3c9c7', '#FFBC42', '#D81159', '#8F2D56', '#218380']
    for i in range(5):
        plt.plot(x, ys[i], colors[i], linewidth=linewidth, linestyle='--' if i==0 else linestyle, marker=markers[i], ms=markersize, label=models[i], mec=mec)
    #l_w2v = plt.plot(x, y_w2v, color='#9baec8', linewidth=linewidth, linestyle=linestyle, marker='o', ms=markersize, label='Word2vec', mec=mec)
    #l_w2v_s = plt.plot(x, y_w2v_s, color='#ffc952', linewidth=linewidth, linestyle=linestyle, marker='v', ms=markersize, label='Word2vec+SWE', mec=mec)
    #l_w2v_r = plt.plot(x, y_w2v_r, color='#47b8e0', linewidth=linewidth, linestyle=linestyle, marker='s', ms=markersize, label='Word2vec+RWE', mec=mec)
    #l_w2v_sr = plt.plot(x, y_w2v_sr, color='#ff7473', linewidth=linewidth, linestyle=linestyle, marker='D', ms=markersize, label='Word2vec+SRWE', mec=mec)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::1], labels[::1])
    plt.xlabel('Dimension')
    plt.ylabel('MSE')
    plt.title('100k-Dataset Experiment Result')
    plt.grid(True, color='#BDBDBD')
    filename = '%s.png' % filename
    plt.savefig(filename, format='png')

if __name__ == '__main__':
    #plot_small_dataset()
    plot_medium_dataset()
