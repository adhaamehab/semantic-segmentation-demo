# -*- coding: utf-8 -*-
import os

import numpy as np

import flask
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import base64


# Setup the app
app = dash.Dash(__name__, static_folder='./static/')
server = app.server
STATIC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

default_img = STATIC_PATH + '/demo.png'
print(default_img)
encoded_image = base64.b64encode(open(default_img, 'rb').read())

app.layout = html.Div([
    html.Div([
        html.H1(
            children="Semantic Segmentation Live Demo",
            style={
                'textAlign': 'left',
                'color': '#b52b3b'
            }
        ),
        html.H3(
            """
            Image segmentation using tensorflow

            This is a retrained model based on tensorflow pre-trained model.

            uses Xception_65 as backbone network and Image net as backbone dataset
            """,
            style={
                'textAlign': 'left',
                'color': '#b52b3b'
            }
        )
    ]),

    html.Hr(),

    html.Div([
        dcc.Input(value='', type='text', placeholder="Enter image url"),
        html.Button("Hit", id='predict')

    ]),

    html.Hr(),

    html.Div([
        html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()))
    ]),

    html.Hr(),

    html.Div([
        html.P(
            'A detailed article about this demo and the backbone model will be published at towardsdatascience.'
        ),
        html.P([
            html.P(children=['You can also view the full source code ', html.A(href="https://github.com/adhaamehab/scene-segmentation", children="here")]),
            html.P(children=['Or visit the author site ', html.A(href="https://adhaamehab.me", children="here")]),

        ])
    ])
])

if __name__ == "__main__":
    app.run_server(debug=True)
