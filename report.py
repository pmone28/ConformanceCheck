def generate_html_report(results_dict, output_file="conformance_report.html"):
    rows = ""
    total = sum(results_dict.values())

    for k, v in results_dict.items():
        percent = (v / total) * 100 if total else 0
        rows += f"<tr><td>{k}</td><td>{v}</td><td>{percent:.2f}%</td></tr>"

    html = f"""
    <html>
    <head>
        <title>Conformance Report</title>
        <style>
            body {{ font-family: Arial; margin: 40px; }}
            h1 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 60%; }}
            th, td {{ border: 1px solid #ccc; padding: 10px; text-align: center; }}
            th {{ background-color: #f4f4f4; }}
            img {{ margin-top: 20px; width: 500px; }}
        </style>
    </head>
    <body>
        <h1>Requirements Conformance Report</h1>

        <h2>Distribution Summary</h2>
        <table>
            <tr><th>Rule</th><th>Violations</th><th>Percentage</th></tr>
            {rows}
        </table>

        <h2>Visual Distribution</h2>
        <img src="conformance_octagonal_pie.png" />

    </body>
    </html>
    """

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"HTML report generated: {output_file}")