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

    fig = tools.make_subplots(rows=2, cols=1, subplot_titles=('Vehicle Densities', 'Vehicle Flows'))

    fig.append_trace(densities, 1, 1)
    fig.append_trace(flow, 2, 1)

    fig['layout']['xaxis1'].update(title='Timestamp (s)')
    fig['layout']['xaxis2'].update(title='Timestamp (s)')

    fig['layout']['yaxis1'].update(title='Number of Vehicles')
    fig['layout']['yaxis2'].update(title='Vehicles Leaving')

    fig['layout'].update(showlegend=False)

    fig['layout'].update(height=600, width=600, title='Vehicle Logs')
    py.offline.plot(fig)
