from flask import Flask, send_from_directory
import os

app = Flask(__name__)


@app.route("/evidence/<category>/<filename>")
def evidence(category, filename):
    return send_from_directory(
        f"evidence/{category}",
        filename
    )


@app.route("/")
def home():

    try:
        with open("logs/alerts.txt", "r") as file:
            logs = file.readlines()
    except:
        logs = []

    garbage = sum("GARBAGE" in log for log in logs)
    trespassing = sum("TRESPASSING" in log for log in logs)
    noparking = sum("NO PARKING" in log for log in logs)

    # Latest evidence images

    try:
        garbage_images = sorted(
            os.listdir("evidence/garbage"),
            reverse=True
        )[:3]
    except:
        garbage_images = []

    try:
        trespassing_images = sorted(
            os.listdir("evidence/trespassing"),
            reverse=True
        )[:3]
    except:
        trespassing_images = []

    try:
        noparking_images = sorted(
            os.listdir("evidence/noparking"),
            reverse=True
        )[:3]
    except:
        noparking_images = []

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>

    <meta http-equiv="refresh" content="5">

    <title>AI Civic Monitoring Dashboard</title>

    <style>

    body {{
        font-family: Arial;
        background-color: #f4f4f4;
        padding: 20px;
    }}

    h1 {{
        text-align: center;
        color: #333;
    }}

    .container {{
        display: flex;
        justify-content: space-around;
        margin-top: 30px;
    }}

    .card {{
        width: 250px;
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
    }}

    .garbage {{
        background: orange;
    }}

    .trespassing {{
        background: crimson;
    }}

    .noparking {{
        background: royalblue;
    }}

    .logs {{
        margin-top: 40px;
        background: white;
        padding: 20px;
        border-radius: 10px;
        max-height: 300px;
        overflow-y: auto;
    }}

    .images {{
        margin-top: 40px;
        background: white;
        padding: 20px;
        border-radius: 10px;
    }}

    img {{
        margin: 10px;
        border-radius: 10px;
        border: 2px solid #ccc;
    }}

    .image-card {{
        display: inline-block;
        margin: 10px;
        text-align: center;
    }}

    .image-card img {{
        width: 250px;
        border-radius: 10px;
        border: 2px solid #ddd;
        transition: 0.3s;
    }}

    .image-card img:hover {{
        transform: scale(1.05);
    }}

    pre {{
        font-size: 15px;
    }}

    </style>

    </head>

    <body>

    <h1>AI Civic Monitoring Dashboard</h1>

    <div class="container">

        <div class="card garbage">
            Garbage Alerts
            <br><br>
            {garbage}
        </div>

        <div class="card trespassing">
            Trespassing Alerts
            <br><br>
            {trespassing}
        </div>

        <div class="card noparking">
            No Parking Alerts
            <br><br>
            {noparking}
        </div>

    </div>
    <div class="logs">

        <h2>Violation Statistics</h2>

        <p>Garbage Alerts: {garbage}</p>
        <p>Trespassing Alerts: {trespassing}</p>
        <p>No Parking Alerts: {noparking}</p>

    </div>

    <div class="logs">

        <h2>Recent Alerts</h2>


        <pre>
{''.join(logs[-20:])}
        </pre>

    </div>

    <div class="images">

        <h2>Latest Evidence Images</h2>

        <h3>Garbage</h3>

        GARBAGE_IMAGES_PLACEHOLDER

        <h3>Trespassing</h3>

        TRESPASS_IMAGES_PLACEHOLDER

        <h3>No Parking</h3>

        NOPARK_IMAGES_PLACEHOLDER

    </div>

    <hr>

    <center>
        <h3>AI Civic Monitoring System</h3>
        <p>Major Project 2025-26</p>
    </center>

    </body>
    </html>
    """

    # Generate Garbage Images HTML

    garbage_html = ""

    for img in garbage_images:
        garbage_html += f"""
        <div class="image-card">
            <img src="/evidence/garbage/{img}">
            <br>
            <small>{img}</small>
        </div>
        """

    # Generate Trespassing Images HTML

    trespass_html = ""

    for img in trespassing_images:
        trespass_html += f"""
        <div class="image-card">
            <img src="/evidence/trespassing/{img}">
            <br>
            <small>{img}</small>
        </div>
        """

    # Generate No Parking Images HTML

    nopark_html = ""

    for img in noparking_images:
       nopark_html += f"""
        <div class="image-card">
            <img src="/evidence/noparking/{img}">
            <br>
            <small>{img}</small>
        </div>
        """

    html = html.replace(
        "GARBAGE_IMAGES_PLACEHOLDER",
        garbage_html
    )

    html = html.replace(
        "TRESPASS_IMAGES_PLACEHOLDER",
        trespass_html
    )

    html = html.replace(
        "NOPARK_IMAGES_PLACEHOLDER",
        nopark_html
    )

    return html


if __name__ == "__main__":
    app.run(debug=True)