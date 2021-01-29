import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def show_heat_maps(*grids,annotate=False,save=None , rc=None,figsize=(30,10),cbar_location='left'):
    fig = plt.figure(figsize=figsize)
    if rc is  None : rc=(1,len(grids))
    grid = ImageGrid(fig, 111, nrows_ncols=rc,axes_pad=0.25,share_all=True,cbar_location=cbar_location,cbar_mode="single",cbar_size="5%",cbar_pad=0.25,)
    for ax,g in zip(grid,grids):
        im = ax.imshow(g[0]) ; 
        ax.title.set_text(g[1])
        if annotate:
            N,M = int(g[0].shape[0]),int(g[0].shape[1])
            for k in range(N):
                for j in range(M):
                    text1 = ax.text(j, k, np.round(g[0][k, j],1),ha="center", va="center", color="w",fontsize=20)


        ax.cax.colorbar(im)
        ax.cax.toggle_label(True)
    
    if save is not None : plt.savefig(save)
    plt.show()
