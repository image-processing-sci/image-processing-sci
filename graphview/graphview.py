#!/usr/bin/env python2.6

from flask import Flask, render_template, request, redirect, session, abort

# from bokeh.embed import components
# from bokeh.plotting import figure
# from bokeh.resources import INLINE
# from bokeh.util.string import encode_utf8

# TODO: create a class to manage all the different graphs
from graphutils.graph3DSurface import Surface3d
from graphutils.graphLine import LineGraph

app = Flask(__name__)

@app.route("/")
def graphview():
    x = [0, 1, 2, 3, 4]
    y = [0, 1, 2, 3, 4]
    z = [5, 6, 7, 8, 9]

    graph_3dsurface, xyz = Surface3d().draw(x, y, z)

    linegraph = LineGraph()
    linegraph.drawLine(x, y)
    graph_lines = linegraph.draw()

    # args = flask.request.args
    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()

    script_3d, div_3d = components(graph_3dsurface)
    script_line, div_line = components(graph_lines)

    html = render_template(
        'graphview.html',
        plot3d_script=script_3d,
        plot3d_div=div_3d,
        line_div=div_line,
        line_script=script_line,
        values2d=dict(zip(x, y)),
        values3d=xyz,
        js_resources=js_resources,
        css_resources=css_resources)

    return encode_utf8(html)

if __name__ == "__main__":
    app.run()
