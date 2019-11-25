import web
import json, os
import entity

urls = (
    '/', 'Root',
    '/process', 'SingleEntityProcessor',
    '/process_all', 'MultiEntityProcessor'
)

app = web.application(urls, globals())

class Root:
    def GET(self):
        return "Generic Entity Processor is up and running..."

class SingleEntityProcessor:
    def POST(self):
        data = json.loads(web.data())
        #text = data['text']
        sentence = data
        result = entity.process(sentence)
        return json.dumps(result)

class MultiEntityProcessor:
    def POST(self):
        data = json.loads(web.data())
        sentences = data['sentences']
        #print("Processing text: {}".format(sentences))
        result = entity.process_all(sentences)
        #print("\nResult:")
        #print(json.dumps(result, indent=2))
        return json.dumps(result)

if __name__ == "__main__":
    app.run()
