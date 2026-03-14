import plotly.express as px

def plot_interactive_html(results_dict):
    labels = list(results_dict.keys())
    values = list(results_dict.values())

    fig = px.pie(
        names=labels,
        values=values,
        title="Requirements Conformance Distribution"
    )

    fig.write_html("interactive_conformance_report.html")
    print("Interactive HTML report saved.")