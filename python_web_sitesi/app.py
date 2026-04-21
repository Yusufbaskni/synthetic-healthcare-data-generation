from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def ana_sayfa():
    return render_template("index.html", baslik="Ana Sayfa")


@app.route("/hakkimda")
def hakkimda():
    return render_template("hakkimda.html", baslik="Hakkimda")


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
