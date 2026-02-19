from openalea.rsml import rsml2mtg, plot2d
import matplotlib.pyplot as plt

graph = rsml2mtg(
    '/home/loai/Documents/code/RSMLExtraction/RSA_reconstruction/Method/ChronoRoot/temp/graphs/ContLight/rpi15_2020-01-08_17-24/4/Plant3/RSML/TimeStep-1488.rsml')
plot2d(graph, show=True)
plt.show()
