from flask import Flask
from flask import request
import json, os
import entity

def get_env(argname, default=None):
    is_bool = type(default) == bool
    try:
        os_val = os.environ[argname]
        if is_bool:
            return os_val.lower().strip() == "true"
        return os_val
    except KeyError:
        if default is None:
            sys.exit("{} needs to be specified as an environment variable, exiting".format(argname))
        return default
app = Flask(__name__)

HOST = get_env('HOST', '127.0.0.1')
PORT = get_env('PORT', 8080)

@app.route('/', methods=['GET'])
def service_info():
    return "Generic Entity Processor is up and running..."

@app.route('/process', methods=['POST'])
def process():
    data = request.get_json(force=True)
    # data = json.loads(web.data())
    sentence = data
    result = entity.process(sentence)
    return json.dumps(result)

@app.route('/process_all', methods=['POST'])
def process_all():
    data = request.get_json(force=True)
    #data = json.loads(web.data())
    sentences = data['sentences']
    result = entity.process_all(sentences)
    return json.dumps(result)

if __name__ == "__main__":
    app.run(HOST, PORT)
