# backend/App.py

import pandas as pd
from flask import Flask, request, jsonify
import pyarrow.parquet as pq
import boto3
from io import BytesIO
from flask_cors import CORS
import ast

app = Flask(__name__)
CORS(app)

s3 = boto3.client('s3')
bucket_name = 'molecular-and-pharmacological-data'
file_key = 'parquet/compoundClassifications_scraped.parquet'

#==========================================================================================

def read_parquet_from_s3():
    try:
        obj = s3.get_object(Bucket=bucket_name, Key=file_key)
        if 'Body' in obj:
            parquet_file = BytesIO(obj['Body'].read())
            table = pq.read_table(parquet_file)
            df = table.to_pandas()
            print("Columns in DataFrame: ", df.columns)  
            print("DataFrame head: ", df.head()) 
            return df
        else:
            print("Error: 'Body' notfound in S3 object")
    except Exception as e:
        print(f"Failed to read from S3: {str(e)}")
    return pd.DataFrame() 

#==========================================================================================

@app.route('/activities', methods=['GET'])
def get_activities():
    try:
        df = read_parquet_from_s3()
        if df.empty:
            return jsonify({'error': 'No data found in Parquet'}), 500

        activity_counts = {}
        total_count = 0
        for activities_str in df['Activities'].dropna():
            try:
                activities = ast.literal_eval(activities_str)
                for activity in activities:
                    if activity in activity_counts:
                        activity_counts[activity] += 1
                    else:
                        activity_counts[activity] = 1
                    total_count += 1 
            except ValueError:
                continue

        activity_list = [{'activity': k, 'count': v} for k, v in sorted(activity_counts.items())]
        activity_list.append({'activity': 'Total', 'count': total_count})  # total
        return jsonify(activity_list)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
#==========================================================================================    

@app.route('/search', methods=['POST'])
def search():
    content = request.json
    search_term = content.get('searchTerm')
    if not search_term:
        return jsonify({'error': 'No search term provided'}), 400
    try:
        df = read_parquet_from_s3()
        if df.empty:
            return jsonify({'error': 'No data found in Parquet'}), 500

        if search_term.isdigit():
            # CID SEARCH
            cid = int(search_term)
            result = df[df['CID'] == cid]

            if not result.empty:
                compound_name = result['Compound Name'].iloc[0] if 'Compound Name' in result.columns else 'N/A'
                activities = []

                if 'Activities' in result.columns:
                    activities_data = result['Activities'].iloc[0]
                    if isinstance(activities_data, str):
                        try:
                            activities = ast.literal_eval(activities_data)
                        except ValueError:
                            activities = []
                    elif pd.isna(activities_data):
                        activities = []
                        
                response = {
                    'CID': cid,
                    'Compound Name': compound_name,
                    'Activities': activities
                }
                return jsonify(response)
            else:
                return jsonify({'error': 'CID not found'}), 404
        else:
            # COMPOUND NAME SEARCH
            compound_name_results = df[df['Compound Name'].str.contains(search_term, case=False, na=False)]
            
            if not compound_name_results.empty:
                response_data = compound_name_results[['CID', 'Compound Name']].to_dict(orient='records')
                return jsonify(response_data)
            else:
                # ACTIVITY SEARCH
                activity = search_term
                filtered_results = df[df['Activities'].apply(lambda x: activity in ast.literal_eval(x) if isinstance(x, str) else False)]
                if not filtered_results.empty:
                    response_data = filtered_results[['CID', 'Compound Name']].to_dict(orient='records')
                    return jsonify(response_data)
                else:
                    return jsonify({'error': 'No matching compounds or activities found'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Unexpected error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)