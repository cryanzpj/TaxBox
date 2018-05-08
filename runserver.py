"""
This script runs the python_webapp_flask application using a development server.
"""

from os import environ
from python_webapp_flask import app
from Binary_loader import Bin_loader
from taxcode_tfidf_search_script import *
from chatbot import *
from flask import request, render_template


@app.route('/taxcodetfidfsearch', methods=['GET'])
def taxcode_tfidf_search():
    if request.method == 'GET':
        query_string = str(request.args.get('search_box'))
        # cosine_sim_threshold = float(request.args.get('cosine_sim_threshold'))
        top = int(request.args.get('number_display'))
        #results = query_wrapper(query_string, cosine_sim_threshold, top, bin)
        return_code, final_code,results = full_response(query_string, top, bin, thres=0.2)
        return render_template('index.html', results= results.to_dict(orient='index') if final_code == 0 else results,
                               return_code = return_code,final_code = final_code,show=top, search_str=query_string)


if __name__ == '__main__':
    HOST = environ.get('SERVER_HOST', 'localhost')

    bin = Bin_loader()
    try:
        PORT = int(environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, PORT)


query_string = "sdfdsfsd"
return_code, final_code,results = full_response(query_string, 3, bin, thres=0.2)