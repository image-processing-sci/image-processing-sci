import plotly as py
import plotly.graph_objs as go
import pickle
import numpy as np
from scipy import interpolate


def plot_logs():
    try:
        log_attributes = pickle.load(open("log_attributes_finished.p", "rb"))
    except:
        log_attributes = pickle.load(open("log_attributes.p", "rb"))

    flow_y = list(range(0, len(log_attributes["flow_timestamps"])))

    #TODO: Figure out how to add titles to the graphs

    # 2d plot
    densities = go.Scatter(
        x=log_attributes["timestamps"],
        y=log_attributes["num_vehicles"],
        name="Density"
    )

    # 2d plot
    flow = go.Scatter(
        x=log_attributes["flow_timestamps"],
        y=flow_y,
        xaxis='x2',
        yaxis='y2',
        name="Flow"
    )

    # 3d plot processing
    time_vs_offset_vs_speed_data = go.Scatter3d(
        x=log_attributes['timestamps'], y=log_attributes['average_offset'], z=log_attributes['average_speed'],
        marker=dict(
            size=5,
            color=log_attributes['average_speed'],
            colorscale='Viridis',  # choose a colorscale
            opacity=0.8
        ),
        scene='scene1',
        # surfaceaxis=2,
        # surfacecolor='red'
        name="Offset vs Speed"
    )

    time_vs_offset_vs_density_data = go.Scatter3d(
        x=log_attributes['timestamps'], y=log_attributes['average_offset'], z=log_attributes['num_vehicles'],
        marker=dict(
            size=5,
            color=log_attributes['num_vehicles'],
            # colorscale='Viridis',  # choose a colorscale
            opacity=0.8
        ),
        scene='scene2',
        # surfaceaxis=2,
        # surfacecolor='red',
        name="Offset vs Density"
    )

    data = [flow, densities, time_vs_offset_vs_speed_data, time_vs_offset_vs_density_data] #, time_vs_speed_vs_density_data]
    layout = go.Layout(
        xaxis=dict(
            domain=[0, 0.45],
            title='Time (s)'
        ),
        yaxis=dict(
            domain=[0, -0.45],
            title='Number of Flowed Cars'
        ),
        xaxis2=dict(
            domain=[0.55, 1],
            title='Time (s)'
        ),
        yaxis2=dict(
            domain=[0, -0.45],
            anchor='x2',
            title='Density of Cars (160ft)'
        ),
        scene1=dict(
            domain=dict(
                x=[0, 0.45],
                y=[0.45, 0],
            ),
            xaxis=dict(
                title='Time (s)'),
            yaxis=dict(
                title='Offset (ft)'),
            zaxis=dict(
                title='Speed (mph)'),
        ),
        scene2=dict(
            domain=dict(
                x=[-1, -1],
                y=[1, 0],
            ),
            xaxis=dict(
                title='Time (s)'),
            yaxis=dict(
                title='Offset (ft)'),
            zaxis=dict(
                title='Density (vehicles/160 ft)'),
        )
    )
    fig = go.Figure(data=data, layout=layout)
    py.offline.plot(fig, filename='density_offset_graphs.html')

    plot3d('timestamps', 'average_offset', 'average_speed', log_attributes, 'speed_vs_offset')
    plot3d('timestamps', 'num_vehicles', 'average_speed', log_attributes, 'speed_vs_density')


def plot3d(x_label, y_label, z_label, log_attributes, outputname):
    x = log_attributes[x_label]
    y = log_attributes[y_label]
    z = log_attributes[z_label]
    # import ipdb; ipdb.set_trace()
    x = np.round(np.asarray(x))
    y = np.round(np.asarray(y))
    z = np.asarray(z)
    z_data_func = interpolate.interp2d(x, y, z, kind='cubic')
    xnew = np.arange(0, max(x), 1)
    ynew = np.arange(0, max(y), 1)
    znew = z_data_func(xnew, ynew)
    data = [
        go.Surface(
            z=znew
        )
    ]
    layout = go.Layout(
        title=outputname,
        autosize=False,
        width=500,
        height=500,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        ),
        scene=dict(
            xaxis=dict(
                title=x_label
            ),
            yaxis=dict(
                title=y_label
            ),
            zaxis = dict(
                title=z_label
            )
        )
    )
    time_vs_offset_vs_speed_data = go.Figure(data=data, layout=layout)
    # py.offline.plot(time_vs_offset_vs_speed_data)
    py.offline.plot(time_vs_offset_vs_speed_data, filename='{0}.html'.format(outputname))


def main():
    pass


if __name__ == '__main__':
    plot_logs()
