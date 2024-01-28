from urllib.parse import urlparse

import cv2
from flask import Flask, request
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


class PeopleCounter(Resource):
    def get(self):
        img = cv2.imread('rynek-Wilkowyje.jpg')
        boxes, weights = hog.detectMultiScale(img, winStride=(4, 4))

        return {'count': len(boxes)}


class PeopleCounterURL(Resource):
    def get(self):
        url = request.args.get('url')

        if url and urlparse(url).scheme and urlparse(url).netloc:
            img = cv2.imread(url)
            if img is not None:
                boxes, weights = hog.detectMultiScale(img, winStride=(4, 4))

                return {'count': len(boxes)}

        return {'error': 'Podany URL nie jest poprawny lub brak parametru "url" w zapytaniu.'}


class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}


api.add_resource(PeopleCounterURL, '/licz')
api.add_resource(PeopleCounter, '/')
api.add_resource(HelloWorld, '/test')

if __name__ == '__main__':
    app.run(debug=True)
