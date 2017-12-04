import plotly as py
import plotly.graph_objs as go
import pickle

def plot_logs():
    try:
        log_attributes = pickle.load(open("log_attributes_finished.p", "rb"))
    except:
        log_attributes = pickle.load(open("log_attributes.p", "rb"))

    flow_y = list(range(0, len(log_attributes["flow_timestamps"])))

    # 2d plot
    densities = go.Scatter(
        x=log_attributes["timestamps"],
        y=log_attributes["num_vehicles"],
    )

    # 2d plot
    flow = go.Scatter(
        x=log_attributes["flow_timestamps"],
        y=flow_y,
        xaxis='x2',
        yaxis='y2'
    )

    # 3d plot processing
    time_vs_density_vs_offset_data = go.Scatter3d(
        x=log_attributes['timestamps'], y=log_attributes['average_offset'], z=log_attributes['average_speed'],
        marker=dict(
            size=5,
            color=log_attributes['average_speed'],
            colorscale='Viridis',   # choose a colorscale
            opacity=0.8
        )
        # surfaceaxis=2,
        # surfacecolor='red'
    )

    data = [flow, densities, time_vs_density_vs_offset_data]
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
        scene=dict(
            domain=dict(
                x=[0, 1],
                y=[1, 0],
            ),
            xaxis=dict(
                title='Time (s)'),
            yaxis=dict(
                title='Offset (ft)'),
            zaxis=dict(
                title='Speed (mph)'),
        )
    )
    fig = go.Figure(data=data, layout=layout)
    py.offline.plot(fig)

def main():
    pass

if __name__ == '__main__':
    plot_logs()