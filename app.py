# -*- coding: utf-8 -*-
import os


import dash
import dash_core_components as dcc
import dash_html_components as html
import base64
from src.model import DeepLabModel
from src.utils import inference

# Setup the app


STATIC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


app = dash.Dash(
    __name__,
    static_folder="./static/",
    assets_external_path="https://fonts.googleapis.com/css?family=Cairo&display=swap",
)

server = app.server

default_img = STATIC_PATH + "/demo.png"

encoded_image = base64.b64encode(open(default_img, "rb").read())

app.layout = html.Div(
    [
        html.Div(
            [
                html.H1(
                    children="Semantic Segmentation Live Demo",
                    style={"textAlign": "center", "color": "#e64659", "size": "30px"},
                ),
                html.P(
                    [
                        html.P("Image segmentation using tensorflow"),
                        html.P(
                            "This is model is a retrained version of deeplab based on tensorflow pre-trained model."
                        ),
                        html.P(
                            "which uses Xception_65 as backbone network and Image net as backbone dataset"
                        ),
                        html.P(
                            children=[
                                "By ",
                                html.A(
                                    href="https://adhaamehab.me", children="Adham Ehab"
                                ),
                            ]
                        ),
                    ],
                    style={"textAlign": "center"},
                ),
            ]
        ),
        html.Hr(),
        html.Div(
            [
                dcc.Input(
                    value="",
                    id="input-box",
                    type="text",
                    placeholder="Enter image url",
                    style={"width": "900px", "height": "40px", "align": "center"},
                ),
                html.Button(
                    "Hit",
                    id="predict",
                    style={
                        "width": "100px",
                        "height": "40px",
                        "background-color": "#e64659",
                        "font-size": "large",
                        "margin": "10px",
                    },
                ),
            ],
            style={"transform": "translate(20%, 5%)"},
        ),
        html.Hr(),
        html.Div(
            [
                html.Img(
                    src="data:image/png;base64,{}".format(encoded_image.decode()),
                    id="result-img",
                )
            ]
        ),
        html.Hr(),
        html.Div(
            [
                html.P(
                    "A detailed article about this demo and the backbone model will be published at towardsdatascience."
                ),
                html.P(
                    [
                        html.P(
                            children=[
                                "You can also view the full source code ",
                                html.A(
                                    href="https://github.com/adhaamehab/scene-segmentation",
                                    children="here",
                                ),
                            ]
                        ),
                        html.P(
                            children=[
                                "Or visit the author site ",
                                html.A(href="https://adhaamehab.me", children="here"),
                            ]
                        ),
                    ]
                ),
            ]
        ),
    ],
    style={"font-family": "Cairo"},
)


@app.callback(
    dash.dependencies.Output("result-img", "src"),
    [dash.dependencies.Input("predict", "n_clicks")],
    [dash.dependencies.State("input-box", "value")],
)
def predict_image(n_clicks, url):
    if n_clicks == 0:
        return "data:image/png;base64,{}".format(encoded_image.decode())
    else:
        try:
            print(url)
            res = "data:image/png;base64,{}".format(
                base64.b64encode(inference(MODEL, url).read()).decode()
            )
            return res
        except Exception as e:
            print(e, url)
            return "data:image/png;base64,{}".format(encoded_image.decode())


if __name__ == "__main__":
    MODEL = DeepLabModel("models/ade20kmodel_may2019.gz")
    print('model downloaded successfully')
    app.run_server(host='0.0.0.0')
