from plotly import tools
import plotly as py
import plotly.graph_objs as go
import pickle

def plot_logs():
    try:
        log_attributes = pickle.load(open("log_attributes_finished.p", "rb"))
    except:
        log_attributes = pickle.load(open("log_attributes.p", "rb"))

    # import ipdb; ipdb.set_trace()

    flow_y = list(range(0, len(log_attributes["flow_timestamps"])))

    densities = go.Scatter(
        x=log_attributes["timestamps"],
        y=log_attributes["num_vehicles"]
    )
    flow = go.Scatter(
        x=log_attributes["flow_timestamps"],
        y=flow_y,
    )

    # 3d plot processing
    time_vs_density_vs_offset_data = go.Scatter3d(
        x = log_attributes['timestamps'], y = log_attributes['average_offset'], z = log_attributes['average_speed'],
        marker=dict(
            size=5,
            color=log_attributes['average_speed'],                # set color to an array/list of desired values
            colorscale='Viridis',   # choose a colorscale
            opacity=0.8
        )
    )
    time_vs_density_vs_offset_data = [time_vs_density_vs_offset_data]
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    time_vs_density_vs_offset = go.Figure(data=time_vs_density_vs_offset_data, layout=layout)

    # diplay all plots

    fig = tools.make_subplots(rows=3, cols=1, subplot_titles=('Vehicle Densities', 'Vehicle Flows'))

    fig.append_trace(densities, 1, 1)
    fig.append_trace(flow, 2, 1)
    fig.append_trace(time_vs_density_vs_offset, 3, 1)


    fig['layout']['xaxis1'].update(title='Timestamp (s)')
    fig['layout']['xaxis2'].update(title='Timestamp (s)')

    fig['layout']['yaxis1'].update(title='Number of Vehicles')
    fig['layout']['yaxis2'].update(title='Vehicles Leaving')

    fig['layout'].update(showlegend=False)

    fig['layout'].update(height=600, width=600, title='Vehicle Logs')
    py.offline.plot(fig)

def get_z_data(log_attributes):
    pass
