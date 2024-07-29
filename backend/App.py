import pandas as pd
from flask import Flask, request, jsonify
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski
import joblib
import requests
from s3_services import s3_client, parquet_handler

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains on all routes


@app.route('/upload', methods=['POST'])
def upload_to_s3():
    # Assume data is sent as a CSV file and converted to Parquet
    df = pd.read_csv(request.files.get('file'))
    parquet_file = 'temp.parquet'
    parquet_handler.write_df_to_parquet(df, parquet_file)
    s3_client.upload_file_to_s3(parquet_file, 'your-bucket-name')
    return jsonify({'message': 'File uploaded successfully'})


# Load the model and the MultiLabelBinarizer
model = joblib.load('C:\\Users\\Lenovo\\Desktop\\Pharmacological-Chemical-Compound-Classifier\\model\\RandomForestModel.pkl')
mlb = joblib.load('C:\\Users\\Lenovo\\Desktop\\Pharmacological-Chemical-Compound-Classifier\\model\\MultiLabelBinarizer.pkl')

app = Flask(__name__)


def fetch_smiles(pubchem_cid):
    url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{pubchem_cid}/property/CanonicalSMILES/JSON'
    response = requests.get(url)
    data = response.json()
    if 'PropertyTable' in data and 'Properties' in data['PropertyTable'] and len(data['PropertyTable']['Properties']) > 0:
        return data['PropertyTable']['Properties'][0]['CanonicalSMILES']
    else:
        return None


def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Add all descriptor computations here, similar to your existing setup
    descriptors = {}  # Dictionary of computed descriptors
    # Populate descriptors dictionary...
    return descriptors


@app.route('/predict', methods=['POST'])
def predict():
    content = request.json
    cid = content['cid']
    smiles = fetch_smiles(cid)
    if smiles:
        descriptors = compute_descriptors(smiles)
        if descriptors:
            df = pd.DataFrame([descriptors])
            prediction = model.predict(df)
            predicted_labels = mlb.inverse_transform(prediction)
            return jsonify({'CID': cid, 'Predicted Activities': predicted_labels})
        else:
            return jsonify({'error': 'Could not compute descriptors from SMILES.'}), 400
    else:
        return jsonify({'error': 'SMILES could not be fetched with given CID.'}), 404


if __name__ == '__main__':
    app.run(debug=True)
