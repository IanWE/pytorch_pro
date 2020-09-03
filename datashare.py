from flask import Flask, send_from_directory

app = Flask(__name__)

@app.route("/download/<fileName>")
def index(fileName):
    return send_from_directory(r"path",filename=fileName,as_attachment=True)

if __name__ == '__main__':
    print(app.url_map)
    app.run(host="202.112.50.151", port=9999)

# for sharing data
