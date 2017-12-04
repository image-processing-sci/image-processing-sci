from plotly import tools
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
        y=log_attributes["num_vehicles"]
    )
    layout1 = dict(domain=[0,10])

    # 2d plot
    flow = go.Scatter(
        x=log_attributes["flow_timestamps"],
        y=flow_y,
    )
    layout2 = dict(domain=[0,10])

    # 3d plot processing
    time_vs_density_vs_offset_data = go.Scatter3d(
        x = log_attributes['timestamps'], y = log_attributes['average_offset'], z = log_attributes['average_speed'],
        marker=dict(
            size=5,
            color=log_attributes['average_speed'],                # set color to an array/list of desired values
            colorscale='Viridis',   # choose a colorscale
            opacity=0.8
        ),
        # surfaceaxis=2,
        # surfacecolor='red'
    )
    import ipdb; ipdb.set_trace()
    # time_vs_density_vs_offset_data = [time_vs_density_vs_offset_data]
    layout3 = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    time_vs_density_vs_offset = go.Figure(data=[time_vs_density_vs_offset_data], layout=layout3)
    data = [flow, densities,time_vs_density_vs_offset_data]
    # layout = [layout1, layout2, layout3]
    layout = go.Layout(
        xaxis=dict(
            domain=[0, 0.45]
        ),
        yaxis=dict(
            domain=[0, 0.45]
        ),
        xaxis2=dict(
            domain=[0.55, 1]
        ),
        yaxis2=dict(
            domain=[0, 0.45],
            anchor='x2'
        ),
        scene=dict(
            domain=dict(
                x=[0.55,1],
                y=[0,1]
            )
        )

    )
    # layout =
    # diplay all plots

    # fig = tools.make_subplots(rows=3, cols=1,
    #         subplot_titles=('Vehicle Densities', 'Vehicle Flows', 'time_vs_density_vs_offset'),
    #         specs=[
    #             [{'is_3d': False}],
    #             [{'is_3d': False}],
    #             [{'is_3d': True}]
    #         ])
    # # fig2 =
    # import ipdb; ipdb.set_trace()

    # fig.append_trace(densities, 1, 1)
    # fig.append_trace(flow, 2, 1)
    # fig.append_trace(time_vs_density_vs_offset, 3, 1)


    # fig['layout']['xaxis1'].update(title='Timestamp (s)')
    # fig['layout']['xaxis2'].update(title='Timestamp (s)')

    # fig['layout']['yaxis1'].update(title='Number of Vehicles')
    # fig['layout']['yaxis2'].update(title='Vehicles Leaving')



    # fig['layout'].update(showlegend=False)

    # fig['layout'].update(height=600, width=600, title='Vehicle Logs')
    fig = go.Figure(data=data, layout = layout)
    py.offline.plot(fig)

def main():
    pass

if __name__ == '__main__':
    plot_logs()