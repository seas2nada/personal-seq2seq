import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# plot attention graph
def PlotAttention(batch_xs, batch_ys, num_seq, attention_graph, graph_dir, epoch='TEST'):
    fig = plt.figure(figsize=(50,50), dpi=100) # for high resolution, use high dpi
    ax = fig.add_subplot(111)
    cax = ax.matshow(attention_graph.cpu().detach().numpy())
    fig.colorbar(cax)
    ax.set_xticklabels(['']+[int(batch_xs[-1,x,-1].cpu().detach().tolist()) for x in range(num_seq*28)])
    ax.set_yticklabels(['']+[int(batch_ys[-1,x,-1].cpu().detach().tolist()) for x in range(num_seq+1)])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(28))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.savefig(graph_dir+'/epoch_'+str(epoch)+'.png')
    plt.close(fig)
    return
