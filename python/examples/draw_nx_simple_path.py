import matplotlib.pyplot as plt
import networkx as nx

G = nx.path_graph(8)
nx.draw_networkx(G)
plt.show()
