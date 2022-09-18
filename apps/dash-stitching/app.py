import numpy as np
import pandas as pd
from skimage import io, data, transform
from time import sleep
import itertools
import base64
import datetime
import io
import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table
import pandas as pd
import dash_bootstrap_components as dbc

import dash
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import dash_table

import dash_canvas
from dash_canvas.components import image_upload_zone
from dash_canvas.utils import (
    image_string_to_PILImage,
    array_to_data_url,
    parse_jsonstring_line,
    brightness_adjust,
    contrast_adjust,
)
from registration import register_tiles
from utils import StaticUrlPath
import pathlib
from function import averageWtp,probability,product_infor,fun_readValuation,profit_rs,price_rs,share_rs,cal_function,pricePE,sharePE,profitPE,profit_ex,price_ex,share_ex,fun_summary_results,valuation,fun_Method_Price_Experiment,cost
from PIL import Image
pil_image = Image.open("/Users/chenyan/Documents/GitHub/dash-sample-apps/apps/dash-stitching/assets/NUS-SIA.png")
pil_image1 = Image.open("/Users/chenyan/Documents/GitHub/dash-sample-apps/apps/dash-stitching/assets/Ancillary.png")
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(
    __name__, external_stylesheets=[dbc.themes.BOOTSTRAP]
)

image1 = Image.open("/Users/chenyan/Documents/GitHub/dash-sample-apps/apps/dash-stitching/static/market.png")

# [dbc.themes.BOOTSTRAP]
# external_stylesheets=external_stylesheets
server = app.server
app.config.suppress_callback_exceptions = True

# get relative data folder
PATH = pathlib.Path(__file__).parent

DATA_PATH = PATH.joinpath("data").resolve()

product_table = pd.DataFrame()
product_table = product_infor()
userPrice=np.zeros(9)
share_mdm = np.zeros(9)
price_mdm = np.zeros(9)
profit_mdm = np.zeros(9)
numberOfExperiments = 30
wtp=np.array([206.47,40.83,24.67,21.02,20.78,13.85,11.84,9.98,5.19])
pricePE,sharePE,profitPE,profit_ex,price_ex,share_ex=fun_Method_Price_Experiment(numberOfExperiments,valuation,wtp,cost)
wtp=np.array([206.47,40.83,24.67,21.02,20.78,13.85,11.84,9.98,5.19])
name_list =[i+1 for i in range(len(profitPE))];
num_list=np.zeros(len(profitPE))
for i in range (len(profitPE)):
    num_list[i]=profitPE[i]


product_list = ['Fare', 'Luggage', 'NO AP','Refund', 'Miles', 'Standby', 'Meal' ,'Boarding', 'Seat']

rowsep = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(html.Div("Demand Distribution")),
            ]
        ),
    ],style={'backgroundColor':'#ef7511','textAlign': 'center', 'color':'white'},
)

rowsep2 = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(html.Div(" Human Decision."),style={'marginLeft': 4}),
                dbc.Col(html.Div(" Mathmatical Models"),style={'marginLeft': 62}),
                dbc.Col(html.Div("Optimazition."),style={'marginLeft': 112}),
            ]
        ),
    ],style={'margin-left' : '30px','textAlign': 'center', 'color':'grey'},
)

Image_intro = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(html.Div(html.Img(src=Image.open(("/Users/chenyan/Documents/GitHub/dash-sample-apps/apps/dash-stitching/static/result.png")),style={'width': '20%', 'height': '30%','margin-top': '16px'}))),
                html.Br(),
                html.Br(),
                dbc.Col(html.Div(html.Img(src=Image.open(("/Users/chenyan/Documents/GitHub/dash-sample-apps/apps/dash-stitching/static/profit.png")),style={'width': '20%', 'height': '30%','margin-top': '16px'}))),
                html.Br(),
                html.Br(),
                html.Br(),
                html.Br(),
                # 
                dbc.Col(html.Div(html.Img(src=Image.open(("/Users/chenyan/Documents/GitHub/dash-sample-apps/apps/dash-stitching/static/price.png")),style={'width': '20%', 'height': '30%','margin-top': '16px'}))),
            ]
        ),
        dbc.Row([rowsep2]),
    ],style={'marginLeft': 122},
)





def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        dash_table.DataTable(
            df.to_dict('records'),
            [{'name': i, 'id': i} for i in df.columns]
        ),

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])

@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

def demo_explanation():
    # Markdown files
    with open(PATH.joinpath("demo.md"), "r") as file:
        demo_md = file.read()

    return html.Div(
        html.Div([dcc.Markdown(demo_md, className="markdown")]),
        style={"margin": "10px"},
    )


def tile_images(list_of_images, n_rows, n_cols):
    dtype = list_of_images[0].dtype
    if len(list_of_images) < n_rows * n_cols:
        white = np.zeros(list_of_images[0].shape, dtype=dtype)
        n_missing = n_rows * n_cols - len(list_of_images)
        list_of_images += [white] * n_missing
    return np.vstack(
        [
            np.hstack(list_of_images[i_row * n_cols : i_row * n_cols + n_cols])
            for i_row in range(n_rows)
        ]
    )


def untile_images(image_string, n_rows, n_cols):
    big_im = np.asarray(image_string_to_PILImage(image_string))
    tiles = [np.split(im, n_cols, axis=1) for im in np.split(big_im, n_rows)]
    return np.array(tiles)


def demo_data():
    im = data.immunohistochemistry()
    l_c = 128
    l_r = 180
    n_rows = 1
    n_cols = im.shape[1] // l_c
    init_i, init_j = 0, 0
    overlap_h = [5, 25]

    big_im = np.empty((n_rows * l_r, n_cols * l_c, 3), dtype=im.dtype)
    i = 0
    for j in range(n_cols):
        sub_im = im[init_i : init_i + l_r, init_j : init_j + l_c]
        big_im[i * l_r : (i + 1) * l_r, j * l_c : (j + 1) * l_c] = sub_im
        init_j += l_c - overlap_h[1]
        init_i += overlap_h[0]
    return big_im


def _sort_props_lines(props, height, width, ncols):
    props = pd.DataFrame(props)
    index_init = ncols * (((props["top"]) - (props["height"]) // 2) // height) + (
        ((props["left"]) - (props["width"]) // 2) // width
    )
    index_end = ncols * (((props["top"]) + (props["height"]) // 2) // height) + (
        ((props["left"]) + (props["width"]) // 2) // width
    )
    props["index_init"] = index_init
    props["index_end"] = index_end
    overlaps = {}
    for line in props.iterrows():
        overlaps[(line[1]["index_init"], line[1]["index_end"])] = (
            int(line[1]["height"]),
            int(line[1]["width"]),
        )
    return overlaps


def instructions():
    return html.P(
        children=[
            """
    When we do pricing, the key issue is 
    that we do not know the demand function. 
    But we can observe the market share from price experiment. 
    We aim to find the optimal prices to maximize total profit, 
    after learning from these pricing experiments
    """
        ],
        className="instructions-sidebar",
    )


accordion = html.Div(
    dbc.Accordion(
        [
            dbc.AccordionItem(
                "This is the content of the first section", title="Type 1"
            ),
            dbc.AccordionItem(
                "This is the content of the second section", title="Type 2"
            ),
            dbc.AccordionItem(
                "This is the content of the third section", title="Type 3"
            ),
        ],
        start_collapsed=True,
        style={'font-size': '10px','width': '330px', 'height': '120px','margin-left': '53px','textAlign': 'left',},
    ),
)

top_card = dbc.Card(
    [
        dbc.CardImg(src=pil_image1, top=True),
        html.Br(),
        dbc.CardBody(
            html.P("Product Structure 1", className="card-text")
        ),
    ],
    style={"width": "18rem",'margin-left': '53px','width': '330px', 'height': '120px','margin-top': '16px','textAlign': 'center',},
)

cards = dbc.Row(
    [
        dbc.Col(top_card, width="auto"),
    ]
)

height, width = 200, 500
canvas_width = 800
canvas_height = round(height * canvas_width / width)
scale = canvas_width / width

list_columns = ["length", "width", "height", "left", "top"]
columns = [{"name": i, "id": i} for i in list_columns]

app.layout = html.Div(
    children=[
        html.Div(
            [
                html.Img(src=pil_image, className="plotly-logo"),
                html.Br(),
                html.Br(),
                html.H1(children=" Market Share Engine"),
                instructions(),
                html.Div(
                    [
                        html.Button(
                            "LEARN MORE",
                            className="button_instruction",
                            id="learn-more-button",
                            style={'width': '300px', 'height': '50%','textAlign': 'center', 'margin-bottom': '30px'}
                        ),
                    ],
                    className="mobile_buttons",
                ),
                html.Div(
                    # Empty child function for the callback
                    html.Div(id="demo-explanation", children=[])
                ),

                html.Div([dbc.Row([dbc.Col(html.Div(" Select Your Product "),style={'marginLeft': 2}),]),],style={'backgroundColor':'#08193b','textAlign': 'center', 'color':'white'},),

                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Number Of Products"),
                                dcc.Input(
                                    id="nrows-stitch",
                                    type="number",
                                    value=1,
                                    name="number of rows",
                                    min=1,
                                    step=1,
                                    style={'width': '330px', 'height': '40%', 'margin-bottom': '30px','margin-left': '52px'}
                                ),
                            ]
                        ),
                        html.Label(" Products Relationships "),
                        accordion,
                        html.Div([
                            html.Label("Constraints Of Products"),
                            dcc.Dropdown(
                                    ['C1', 'C2', 'C3'],
                                    ['C1', 'C2'],
                                    multi=True,
                                    style={'width': '330px', 'height': '30%', 'margin-bottom': '20px','margin-left': '28px'}
                                )
                         ]),
                    ],
                    className="mobile_forms",
                ),
                html.Div(
                    [
                        html.Label("Products Structure "),
                        # html.Br(),
                        cards,
                    ],
                    className="radio_items",
                ),
                html.Br(),
                html.Button(
                    "Run Calculation", id="button-stitch", className="button_submit",style={'width': '310px', 'height': '50%','textAlign': 'center', 'margin-bottom': '30px',}
                ),
                html.Br(),
            ],
            className="four columns instruction",
        ),

        html.Div(
            [
                dcc.Tabs(
                    id="stitching-tabs",
                    value="canvas-tab",
                    children=[
                        dcc.Tab(label="Homepage", value="home-tab",style={'backgroundColor':'#08193b','font-size': '15px'}),
                        dcc.Tab(label="Analysis & Visualization", value="canvas-tab",style={'backgroundColor':'#08193b','font-size': '15px'}),
                        dcc.Tab(label="Engine Simulation", value="result-tab",style={'backgroundColor':'#08193b','font-size': '15px'}),
                        dcc.Tab(label="Engine Optimizer", value="opt-tab",style={'backgroundColor':'#08193b','font-size': '15px'}),
                        dcc.Tab(label="Feedback", value="help-tab",style={'backgroundColor':'#08193b','font-size': '15px'}),
                    ],
                    className="tabs",
                    style={'backgroundColor':'blue','textAlign': 'center', 'color':'white','height': '90px',
    'width': '9200px',"margin": "24px",'margin-top': '30px','font-size': '15px'},
                ),
                html.Div(
                    id="tabs-content-example",
                    className="canvas",
                    style={"text-align": "left", "margin": "auto"},
                ),
                html.Div(className="upload_zone", id="upload-stitch", children=[]),
                html.Div(id="sh_x", hidden=True),
                html.Div(id="stitched-res", hidden=True),
                dcc.Store(id="memory-stitch"),
            ],
            className="eight columns result",
            # style={'backgroundColor':'#ef7511','textAlign': 'center', 'color':'white'},
        ),
    ],
    className="row twelve columns",
)


@app.callback(
    Output("tabs-content-example", "children"), [Input("stitching-tabs", "value")]
)
def fill_tab(tab):
    if tab == "canvas-tab":
        return [
            html.Div(
                children=[
                    html.Div([
                        dcc.Upload(
                            id='upload-data',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select Files')
                            ]),
                            style={
                                'width': '95%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '30px'
                            },
                            # Allow multiple files to be uploaded
                            multiple=True
                        ),
                        html.Div(id='output-data-upload'),
                    ]),
                ],
                className="upload_zone",
                id="upload",
            ),
        ]
    elif tab == "home-tab":
        # return html.Div([home_carousel])
        return [html.Div(id="back",children=[html.Img(src=image1,style={'width': '100%', 'height': '30%'})]),html.Div(Image_intro)]
    elif tab == "opt-tab":
        return [html.P(['We have 9 different products', html.Br(), html.Br(), 'Fare | Luggage | NO AP | Refund | Miles | Standby | Meal | Boarding | Seat '])]
    elif tab == "result-tab":
        return [
                html.Div(
                    id="bottom-column",
                    className="twelve columns",
                    children=[
                        html.Div([
                            html.H3("Input Product Price "),
                            html.Br(),
                            html.Br(),
                            # 1st Col 
                            html.Div([html.Div([html.P(['  Fare '])],style={'width': '70vh', 'height': '8vh'}),html.Div([html.P(['  Luggage'])],style={'width': '70vh', 'height': '9vh'}),html.Div([html.P(['  No AP'])],style={'width': '70vh', 'height': '2vh'}),],className="two columns"),
                            html.Div([html.Div([dcc.Input(
                                id='Property_name',
                                placeholder='Fare',
                                type='text',
                                value='',
                            ),]),
                            html.Br(),
                            html.Div([dcc.Input(
                                id='Stree_name',
                                placeholder='Luggage',
                                type='text',
                                value='',
                            ),]),
                            html.Br(),
                            html.Div([dcc.Input(
                                id='City',
                                placeholder='No AP',
                                type='text',
                                value='',
                            ),]),],className="two columns"),

                            # 2nd col
                            html.Div([html.Div([html.P(['Refund '])],style={'width': '70vh', 'height': '8vh'}),html.Div([html.P(['Miles '])],style={'width': '70vh', 'height': '8vh'}),html.Div([html.P(['Standby'])],style={'width': '70vh', 'height': '8vh'}),],className="two columns"),
                            html.Div([html.Div([dcc.Input(
                                id='Zip_code',
                                placeholder='Refund',
                                type='text',
                                value='',
                            ),]),
                            html.Br(),
                            html.Div([dcc.Input(
                                id='Country',
                                placeholder='Miles',
                                type='text',
                                value='',
                            ),]),
                            html.Br(),
                            html.Div([dcc.Input(
                                id='V6',
                                placeholder='Standby',
                                type='text',
                                value='',
                            ),]),],className="two columns"),

                            # 3rd Col 
                            html.Div([html.Div([html.P(['Meal '])],style={'width': '70vh', 'height': '8vh'}),html.Div([html.P(['Boarding '])],style={'width': '70vh', 'height': '8vh'}),html.Div([html.P(['Seat'])],style={'width': '70vh', 'height': '8vh'}),],className="two columns"),
                            html.Div([
                            html.Div([dcc.Input(
                                id='V7',
                                placeholder='Meal',
                                type='text',
                                value='',
                            ),]),
                            html.Br(),
                            html.Div([dcc.Input(
                                id='V8',
                                placeholder='Boarding',
                                type='text',
                                value='',
                            ),]),
                            html.Br(),
                            html.Div([dcc.Input(
                                id='V9',
                                placeholder='Seat',
                                type='text',
                                value='',
                            ),]),],className="two columns"),
                            
                            
                            
                            html.Br(),
                            html.Br(),
                            html.Br(),
                            html.Br(),
                            html.Br(),
                            html.Div(id='address'),
                        ]),

                        html.Div([html.Br(),html.Button(id='Submit_address', n_clicks=0, children='Submit')],className="twelve columns",style={'margin-left': '550px',
                                 'margin-bottom': '10px',
                                 'verticalAlign': 'middle',
                                 'color':"secondary",}),
                
            ],
        ),
        ]
    # help-Tab 
    return [html.P(['We have 9 different products', html.Br(), html.Br(), 'Fare | Luggage | NO AP | Refund | Miles | Standby | Meal | Boarding | Seat '])]



@app.callback(
    [Output('address', 'children')],
    [Input('Submit_address', 'n_clicks')],
    [State('Property_name', 'value'),
     State('Stree_name', 'value'),
     State('City', 'value'),
     State('Zip_code', 'value'),
     State('Country', 'value')])

def update_map(n_clicks,fare, luggage, NoAP, Refund,Miles):
    inputlis = [list({fare}),list({luggage}),list({NoAP}),list({Refund}),list({Miles}),[0],[1],[1],[1]]
    v1 = list({fare})
    ab = itertools.chain(list({fare}), list({luggage}),list({NoAP}),list({Refund}),list({Miles}),['2'], ['3'],['2'], ['3'])
    ab = list(ab)
    ab = list(map(int, ab))

    # summary 
    summary,sumresult=fun_summary_results(9,price_rs,share_rs,profit_rs,price_ex,share_ex,profit_ex,price_mdm,share_mdm,profit_mdm)
    lst = [['Price - Random Search'],['Share - Random Search'],['Price - Experiment Best'],['Share - Experiment Best'],['Price - MDM'],['Share - MDM']]
    sumresult.insert(loc=0, column='RowName', value=lst)
    df = pd.DataFrame(sumresult, columns =['RowName','Fare', 'Luggage', 'No AP','Refund','Miles','Standby','Meal','Boarding','Seat','Total Profit'])

    result = cal_function(ab,pricePE,sharePE,profitPE)
    return [html.Div(f"User Share: {result[0]}"),html.Div(f"User Profit: {result[1]}"),html.Div(f"price_ex: {result[2]}"),html.Div(f"profit_ex: {result[3]}"),html.Div(f"share_ex: {result[4]}"),html.Div(f"numberOfExperiments: {result[5]}"),html.Div(f" Summary Table : "),dash_table.DataTable(
            id="t1",
            data=df.to_dict('records'),
            columns=[
                {'name': i, 'id': i} for i in df.columns
            ],
            fixed_rows={ 'headers': True, 'data': 0 },
            style_as_list_view=True,
            style_cell={
                'padding': '5px',
                'border': '1px solid black'},
            style_header={
                'backgroundColor': 'white',
                'fontWeight': 'bold',
                'border': '1px solid black'
            },
            virtualization=True,
            page_action='none'
            )],

@app.callback(Output("stitching-tabs", "value"), [Input("button-stitch", "n_clicks")])
def change_focus(click):
    if click:
        return "result-tab"
    return "canvas-tab"


@app.callback(Output("table-stitch", "data"), [Input("canvas-stitch", "json_data")])
def estimate_translation(string):
    props = parse_jsonstring_line(string)
    if props is not None and len(props) > 0:
        df = pd.DataFrame(props, columns=list_columns)
        return df.to_dict("records")
    else:
        raise PreventUpdate


@app.callback(Output("memory-stitch", "data"), [Input("button-stitch", "n_clicks")])
def update_store(click):
    return click


@app.callback(
    Output("stitching-result", "src"),
    [
        Input("contrast-stitch", "value"),
        Input("brightness-stitch", "value"),
        Input("stitched-res", "children"),
    ],
)
def modify_result(contrast, brightness, image_string):
    if image_string is None:
        raise PreventUpdate
    img = np.asarray(image_string_to_PILImage(image_string))
    img = contrast_adjust(img, contrast)
    img = brightness_adjust(img, brightness)
    return array_to_data_url(img)


@app.callback(
    Output("stitched-res", "children"),
    [Input("button-stitch", "n_clicks")],
    [
        State("nrows-stitch", "value"),
        State("ncolumns-stitch", "value"),
        State("overlap-stitch", "value"),
        State("table-stitch", "data"),
        State("sh_x", "children"),
        State("do-blending-stitch", "values"),
    ],
)
def modify_content(n_cl, n_rows, n_cols, overlap, estimate, image_string, vals):
    blending = 0
    if vals is not None:
        blending = 1 in vals
    if image_string is None:
        raise PreventUpdate
    tiles = untile_images(image_string, n_rows, n_cols)
    if estimate is not None and len(estimate) > 0:

        overlap_dict = _sort_props_lines(
            estimate, tiles.shape[2], tiles.shape[3], n_cols
        )
    else:

        overlap_dict = None
    canvas = register_tiles(
        tiles,
        n_rows,
        n_cols,
        overlap_global=overlap,
        overlap_local=overlap_dict,
        pad=np.max(tiles.shape[2:]),
        blending=blending,
    )
    return array_to_data_url(canvas)


@app.callback(Output("canvas-stitch", "image_content"), [Input("sh_x", "children")])
def update_canvas_image(im):
    return im


@app.callback(Output("upload-stitch", "contents"), [Input("demo", "n_clicks")])
def reset_contents(n_clicks):
    if n_clicks:
        return None


@app.callback(
    Output("sh_x", "children"),
    [
        Input("upload-stitch", "contents"),
        Input("upload-stitch", "filename"),
        Input("demo", "n_clicks"),
        Input("downsample", "value"),
    ],
    [State("nrows-stitch", "value"), State("ncolumns-stitch", "value")],
)
def upload_content(
    list_image_string, list_filenames, click, downsample, n_rows, n_cols
):

    downsample = int(downsample)
    if list_image_string is not None:
        order = np.argsort(list_filenames)
        image_list = [
            np.asarray(image_string_to_PILImage(list_image_string[i])) for i in order
        ]
        if downsample > 1:
            ratio = 1.0 / downsample
            multichannel = image_list[0].ndim > 2
            image_list = [
                transform.rescale(
                    image, ratio, multichannel=multichannel, preserve_range=True
                ).astype(np.uint8)
                for image in image_list
            ]
        res = tile_images(image_list, n_rows, n_cols)
        return array_to_data_url(res)
    elif click:
        res = demo_data()
        tmp = array_to_data_url(res)
        return tmp

    raise PreventUpdate


@app.callback(
    [Output("demo-explanation", "children"), Output("learn-more-button", "children")],
    [Input("learn-more-button", "n_clicks")],
)
def learn_more(n_clicks):
    if n_clicks is None:
        n_clicks = 0
    if (n_clicks % 2) == 1:
        n_clicks += 1
        return (
            html.Div(
                className="demo_container",
                style={"margin-bottom": "30px"},
                children=[demo_explanation()],
            ),
            "Close",
        )

    n_clicks += 1
    return (html.Div(), "Learn More")


if __name__ == "__main__":
    app.run_server(debug=True)
