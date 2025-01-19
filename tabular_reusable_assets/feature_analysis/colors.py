import matplotlib.colors as mcolors
import plotly.express as px


# https://stackoverflow.com/questions/77886066/plotly-colormaps-in-matplotlib, plotly colors for matplotlib
SAMPLES = 20
ice = px.colors.sample_colorscale(px.colors.sequential.ice, SAMPLES)
rgb = [px.colors.unconvert_from_RGB_255(px.colors.unlabel_rgb(c)) for c in ice]
cmap = mcolors.ListedColormap(rgb, name="Ice", N=SAMPLES)

plotly_colors = px.colors.qualitative.Plotly
