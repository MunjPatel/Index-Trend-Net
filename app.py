# from flask import Flask, render_template, request, redirect, url_for, jsonify
# import joblib
# import json
# import numpy as np
# import os
# from preprocessing import ProcessTickerData
# import plotly
# import plotly.graph_objs as go

# app = Flask(__name__)

# # Load tickers and countries from JSON
# with open("tickers.json", "r") as f:
#     tickers = json.load(f)

# @app.route('/')
# def home():
#     # Display the home page with a dropdown for tickers and their corresponding countries
#     return render_template('home.html', tickers=tickers)

# @app.route('/select_ticker', methods=['POST'])
# def select_ticker():
#     # Get the selected ticker from the form
#     selected_ticker = request.form.get('ticker')
#     return redirect(url_for('performance', ticker=selected_ticker))

# @app.route('/performance/<ticker>')
# def performance(ticker):
#     # Load the results from the results folder
#     result_path = os.path.join("results", f"{ticker}.json")
#     with open(result_path, "r") as f:
#         results = json.load(f)

#     # Create interactive graphs for y_test vs y_pred
#     y_test = results['y_test'][-30:]  # Last 30 values
#     y_pred = results['y_pred'][-30:]  # Last 30 values

#     # Generate line chart for y_test vs y_pred
#     line_chart = generate_comparison_chart(y_test, y_pred, ticker)

#     # Generate confusion matrix heatmap
#     confusion_matrix = results["confusion_matrix"]
#     heatmap_chart = generate_confusion_matrix_heatmap(confusion_matrix)

#     # Render the performance template with model performance metrics
#     return render_template('performance.html', ticker=ticker, results=results, 
#                            line_chart=line_chart, heatmap_chart=heatmap_chart)

# def generate_comparison_chart(y_test, y_pred, ticker):
#     # Line chart for y_test vs y_pred
#     trace_test = go.Scatter(x=list(range(len(y_test))), y=y_test, mode='lines+markers', name='y_test')
#     trace_pred = go.Scatter(x=list(range(len(y_pred))), y=y_pred, mode='lines+markers', name='y_pred')

#     layout = go.Layout(title=f"{ticker} - Last 30 Days Actual vs Predicted", xaxis=dict(title="Days"), yaxis=dict(title="Value"))
#     fig = go.Figure(data=[trace_test, trace_pred], layout=layout)
#     return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

# def generate_confusion_matrix_heatmap(confusion_matrix):
#     # Confusion matrix heatmap
#     heatmap = go.Heatmap(z=confusion_matrix, x=["Predicted 0", "Predicted 1"], y=["Actual 0", "Actual 1"], colorscale="Viridis")
#     layout = go.Layout(title="Confusion Matrix", xaxis=dict(title="Predicted"), yaxis=dict(title="Actual"))
#     fig = go.Figure(data=[heatmap], layout=layout)
#     return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for, jsonify
import json
import os
import plotly
import plotly.graph_objs as go

app = Flask(__name__)

# Load tickers and countries from JSON
with open("tickers.json", "r") as f:
    tickers = json.load(f)

@app.route('/')
def home():
    # Display the home page with an introduction and a dropdown for tickers
    return render_template('home.html', tickers=tickers)

@app.route('/select_ticker', methods=['POST'])
def select_ticker():
    # Get the selected ticker from the form
    selected_ticker = request.form.get('ticker')
    return redirect(url_for('performance', ticker=selected_ticker))

@app.route('/performance/<ticker>')
def performance(ticker):
    # Load the results from the results folder
    result_path = os.path.join("results", f"{ticker}.json")
    with open(result_path, "r") as f:
        results = json.load(f)

    # Prepare data for the stacked bar chart
    y_test = results['y_test']
    y_pred = results['y_pred']
    
    # Generate stacked bar chart data
    stacked_bar_chart = generate_stacked_bar_chart(y_test, y_pred)

    # Generate confusion matrix heatmap
    confusion_matrix = results["confusion_matrix"]
    heatmap_chart = generate_confusion_matrix_heatmap(confusion_matrix)

    # Render the performance template with model performance metrics
    return render_template('performance.html', ticker=ticker, results=results, 
                           stacked_bar_chart=stacked_bar_chart, heatmap_chart=heatmap_chart)

def generate_stacked_bar_chart(y_test, y_pred):
    # Flatten y_pred as it may be in a nested format
    y_pred = [item[0] if isinstance(item, list) else item for item in y_pred]

    # Calculate counts for each category (0 and 1) in y_test and y_pred
    actual_counts = [y_test.count(0), y_test.count(1)]
    predicted_counts = [y_pred.count(0), y_pred.count(1)]

    # Create stacked bar chart data
    trace_actual = go.Bar(x=['Negative', 'Positive'], y=actual_counts, name='Actual')
    trace_predicted = go.Bar(x=['Negative', 'Positive'], y=predicted_counts, name='Predicted')

    layout = go.Layout(
        title="Actual vs Predicted - Stacked Bar Chart",
        xaxis=dict(title="Category"),
        yaxis=dict(title="Count"),
        barmode="stack"
    )

    fig = go.Figure(data=[trace_actual, trace_predicted], layout=layout)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def generate_confusion_matrix_heatmap(confusion_matrix):
    # Confusion matrix heatmap
    heatmap = go.Heatmap(z=confusion_matrix, x=["Predicted 0", "Predicted 1"], y=["Actual 0", "Actual 1"], colorscale="Viridis")
    layout = go.Layout(title="Confusion Matrix", xaxis=dict(title="Predicted"), yaxis=dict(title="Actual"))
    fig = go.Figure(data=[heatmap], layout=layout)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

if __name__ == '__main__':
    app.run(debug=True)
