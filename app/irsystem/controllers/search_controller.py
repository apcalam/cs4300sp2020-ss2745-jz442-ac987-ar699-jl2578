from . import *
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder

project_name = "Gifter.ai"
net_id = "Shreya Subramanian: ss2745, Joy Zhang: jz442, Aparna Calambur: ac987, Ashrita Raman: ar699, Jannie Li: jl2578"


@irsystem.route('/', methods=['GET'])
def search():
    query = request.args.get('search')
    if not query:
        data = []
        output_message = ''
    else:
        output_message = "Your search: " + query
        data = range(5)
    return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data)
