import plotly
from plotly.graph_objs import Scatter, Layout
import pickle

# import ipdb; ipdb.set_trace()

def plot_densities():
    global_densities = pickle.load(open("global_densities.p", "rb"))

    plotly.offline.plot({
        "data": [Scatter(x=global_densities["timestamps"], y=global_densities["num_vehicles"])],
        "layout": Layout(title="Density of vehicles per hour",
                         xaxis=dict(
                             title='Timestamp (s)',
                             titlefont=dict(
                                 family='Arial, sans-serif',
                                 size=18,
                             ),
                             showticklabels=True,
                             tickangle=45,
                             tickfont=dict(
                                 family='Old Standard TT, serif',
                                 size=14,
                                 color='black'
                             ),
                             exponentformat='e',
                             showexponent='All'
                         ),
                         yaxis=dict(
                             title='Number of Vehicles',
                             titlefont=dict(
                                 family='Arial, sans-serif',
                                 size=18,
                             ),
                             showticklabels=True,
                             tickangle=45,
                             tickfont=dict(
                                 family='Old Standard TT, serif',
                                 size=14,
                                 color='black'
                             ),
                             exponentformat='e',
                             showexponent='All'
                         )
                         )
    })
