import numpy as np
import matplotlib.pyplot as plt
import matplotlib

__author__ = 'calandra'
version = '0.1'


def paretoFront(PF, drawConnectingLines=True, lines='minimize', color='red', marker='o', markersize=6,  linewidth=4, maxNumberMarkers=np.inf, markerOffset=0):
    """
    
    :param PF: [N_OBJECTIVES, N_POINTS]
    :param lines: 
    :param color: 
    :param marker: 
    :param markersize: 
    :param drawConnectingLines: 
    :param linewidth: 
    :param maxNumberMarkers: 
    :param markerOffset: 
    :return: 
    """
    nDim = PF.shape[0]
    nPoints = PF.shape[1]
    assert nDim == 2, 'This function can only plot 2D Pareto fronts'

    if nPoints < maxNumberMarkers:
        # Plot points
        h, = plt.plot(PF[0], PF[1],
                      linestyle='None',
                      marker=marker,
                      fillstyle='full',
                      markersize=markersize,
                      markerfacecolor=color,
                      markeredgecolor=color,
                      rasterized=False,
                      antialiased=True,
                      clip_on=False,
                      )
    else:
        # Equi-distribute markers along the curve
        s_x = np.sort(PF[0])
        idx_curve = np.linspace(s_x[0], stop=s_x[-1], num=maxNumberMarkers)+markerOffset
        if markerOffset > 0:
            idx_curve = idx_curve[0:-1]
        idx_markers = np.zeros(maxNumberMarkers, dtype=int)
        for idx, value in enumerate(idx_curve):
            idx_markers[idx] = (np.abs(PF[0]-value)).argmin()
        # Plot points
        h, = plt.plot(PF[0][idx_markers], PF[1][idx_markers],
                      linestyle='None',
                      marker=marker,
                      fillstyle='full',
                      markersize=markersize,
                      markerfacecolor=color,
                      markeredgecolor=color,
                      rasterized=False,
                      antialiased=True,
                      clip_on=False,
                      )

    # Plot lines
    if drawConnectingLines:
        if lines in ('minimize', 'maximize'):
            # sort PF
            sortedPF = PF[:, np.argsort(PF[0])]
            for line in range(0, nPoints-1):
                if lines == 'minimize':
                    plt.plot([sortedPF[0, line], sortedPF[0, line+1]], [sortedPF[1, line], sortedPF[1, line]], color=color, linewidth=linewidth)
                    plt.plot([sortedPF[0, line+1], sortedPF[0, line+1]], [sortedPF[1, line], sortedPF[1, line+1]], color=color, linewidth=linewidth)
                if lines == 'maximize':
                    plt.plot([sortedPF[0, line], sortedPF[0, line]], [sortedPF[1, line], sortedPF[1, line+1]], color=color, linewidth=linewidth)
                    plt.plot([sortedPF[0, line], sortedPF[0, line+1]], [sortedPF[1, line+1], sortedPF[1, line+1]], color=color, linewidth=linewidth)

    # Create custom artist
    h = matplotlib.lines.Line2D([], [], color=color, linewidth=linewidth, marker=marker, markersize=markersize)

    return h


def paretoFront_webplot():
    # TODO: implement me
    h = []
    return h
