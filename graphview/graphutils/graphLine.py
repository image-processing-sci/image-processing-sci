from bokeh.plotting import figure

class LineGraph():
    """
        LineGraph
        A simple wrapper for creating a plot and drawing lines on it. 
        
        title - title of the graph
        xlabel - x axis label
        ylabel - y axis label
        width - width of the graph
        height - height of the graph
    """
    def __init__(self, title="", xlabel="", ylabel="", width=1000, height=600):
        self.plot = figure(plot_width=width, plot_height=height)
       
    def drawLine(self, xdata, ydata, color='black'):
        self.plot.line(xdata, ydata, line_width=2)
        
    def draw(self):
        return self.plot
