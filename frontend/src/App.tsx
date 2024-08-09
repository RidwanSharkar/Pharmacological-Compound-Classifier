/* frontend/src/App.tsx: */

import React, { useState } from 'react';
import axios from 'axios';
import './App.css';


const FLASK_SERVER_URL = 'http://localhost:5000';

interface Compound {
  CID: number;
  'Compound Name': string;
}

interface ActivityResult {
  CID: number;
  'Compound Name': string;
  Activities: string[];
}

interface PredictionResult {
  CID: number;
  'Compound Name': string;
  'Predicted Activities': string[];
}

interface Activity {
  activity: string;
  count: number;
}

interface SortConfig {
  key: keyof Activity;  // to oscillate
  direction: 'ascending' | 'descending';
}




/*========================================================================================================*/

const App: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [results, setResults] = useState<Compound[] | ActivityResult | PredictionResult | null>(null);
  const [error, setError] = useState('');
  const [isBrowsing, setIsBrowsing] = useState(false);  // Swap Table <-> Search View
  const [sortConfig, setSortConfig] = useState<SortConfig>({ key: 'count', direction: 'descending' }); 
  const [activities, setActivities] = useState<Activity[]>([]);
  const [imageUrl, setImageUrl] = useState('');
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null); 
  
/*========================================================================================================*/

  /* SEPARATE ACTIVITY SEARCH FROM PIC process */
  const handleSearch = async (cid?: string) => {
    const searchId = cid || searchTerm; // FOR CLICKING VS SEARCHING

    if (!searchId.trim()) {
        setError('Please enter a search term.');
        return;
    }

    setIsBrowsing(false);

    try {
        const detailResponse = await axios.post(`${FLASK_SERVER_URL}/search`, { searchTerm: searchId });
        if (detailResponse.data.error) {
            setError(detailResponse.data.error);
            setResults(null);
        } else {
            setResults(detailResponse.data);
            setError('');

            // SCRAPES IMAGE FROM PUBCHEM
            const imageUrl = `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/CID/${searchId}/PNG`;
            invertImageColors(imageUrl).then(setImageUrl).catch((error) => setError('Failed to process image'));
        }
    } catch (error) {
        if (axios.isAxiosError(error)) {
            setError(`Error: ${error.response ? error.response.data.error : 'Server not reachable'}`);
        } else {
            setError('An unexpected error occurred. Please try again.');
        }
        setResults(null);
    }
    const searchUrl = `/results?query=${encodeURIComponent(searchTerm)}`;
    window.history.pushState({}, '', searchUrl);
};

  const fetchActivities = async () => {
    setIsBrowsing(true); // To swap from table
    try {
      const response = await axios.get(`${FLASK_SERVER_URL}/activities`);
      setActivities(response.data);
      setError('');
      setResults(null); 
    } catch (error) {
      setError('Failed to fetch activities');
      setActivities([]);
    }
    window.history.pushState({}, '', '/browse');
  };

  /*========================================================================================================*/
  
  const sortActivities = (a: Activity, b: Activity) => {
    if (a[sortConfig.key] < b[sortConfig.key]) {
      return sortConfig.direction === 'ascending' ? -1 : 1;
    }
    if (a[sortConfig.key] > b[sortConfig.key]) {
      return sortConfig.direction === 'ascending' ? 1 : -1;
    }
    return 0;
  };

  const requestSort = (key: keyof Activity) => {
    let direction: 'ascending' | 'descending' = 'ascending';
    if (sortConfig.key === key && sortConfig.direction === 'ascending') {
      direction = 'descending';
    }
    setSortConfig({ key, direction });
  };

  /*========================================================================================================*/

  const handleActivityClick = async (activity: string) => {
    setIsBrowsing(false); 
    setSearchTerm(activity); 
    try {
      const response = await axios.post(`${FLASK_SERVER_URL}/search`, { searchTerm: activity });
      console.log(response.data); 
      if (response.data.error) {
        setError(response.data.error);
        setResults(null);
      } else {
        setResults(response.data);
        setError('');
      }
    } catch (error) {
      if (axios.isAxiosError(error)) {
        setError(`Error: ${error.response ? error.response.data.error : 'Server not reachable'}`);
      } else {
        setError('An unexpected error occurred. Please try again.');
      }
      setResults(null);
    }
  };

  const handleCIDClick = async (cid: number) => {
    handleSearch(cid.toString());
    window.history.pushState({}, '', `/compound/${cid}`);
};

  /* INVERTER =======================================================================================*/

  const invertImageColors = async (imageSrc: string) => {
    const image = new Image();
    image.crossOrigin = 'Anonymous'; 
    image.src = imageSrc;

    return new Promise<string>((resolve, reject) => {
        image.onload = () => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            if (!ctx) {
                reject('Failed to get canvas context');
                return;
            }

            canvas.width = image.width;
            canvas.height = image.height;
            
            ctx.drawImage(image, 0, 0);
            
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const data = imageData.data;

            for (let i = 0; i < data.length; i += 4) {
              data[i] = 255 - data[i];        // R
              data[i + 1] = 255 - data[i + 1];  // G
              data[i + 2] = 255 - data[i + 2];  // B
            }
            // NOT EXACTLY BLACK AFTER INVERSION (#0a0a0a)
            for (let i = 0; i < data.length; i += 4) {
              if (data[i] < 15 && data[i + 1] < 15 && data[i + 2] < 15) {
                  data[i] = 51;     
                  data[i + 1] = 51; 
                  data[i + 2] = 51; 
              }
          }
            
            ctx.putImageData(imageData, 0, 0);
            resolve(canvas.toDataURL());
        };

        image.onerror = (error) => {
            reject(error);
        };
    });
  };


  /*========================================================================================================*/

  const renderActivities = () => {
    if (!activities.length || !isBrowsing) {
      return null;
    }
  
    const sortedActivities = [...activities].sort(sortActivities);

    return (
      <div>
        <h2> </h2>
        <table style={{ borderRadius: '22px', overflow: 'hidden', }}>
          <thead>
            <tr>
              <th style={{ textAlign: 'center' }}>
                <button 
                  onClick={() => requestSort('activity')}
                  style={{ border: 'none', background: 'none', color: 'white', textAlign: 'center', fontSize: '20px' }}
                >
                  PubChem Pharmacological Classification:
                </button>
              </th>
              <th style={{ width: '50px', textAlign: 'center' }}>
                <button 
                  onClick={() => requestSort('count')}
                  style={{ border: 'none', background: 'none', color: 'white', textAlign: 'center', fontSize: '14px' }}
                >
                  Entries
                </button>
              </th>
            </tr>
          </thead>

          <tbody>
            {sortedActivities.map((item, index) => (
              <tr key={index}
                  onMouseEnter={() => setHoveredIndex(index)}
                  onMouseLeave={() => setHoveredIndex(null)}>
                <td style={{ background: hoveredIndex === index ? '#45a049' : '#575757' }}>
                  <button 
                    onClick={() => handleActivityClick(item.activity)}
                    style={{ textTransform: 'capitalize', border: 'none', background: 'none', fontSize: '15px' }}>
                    {item.activity}
                  </button>
                </td>
                <td style={{ fontFamily: 'Arial, sans-serif', textAlign: 'center', background: hoveredIndex === index ? '#45a049' : '#575757' }}>
                  {item.count}
                </td>
              </tr>
            ))}
          </tbody>


        </table>
      </div>
    );
  };

  /*========================================================================================================*/

  const renderResults = () => {
    if (!results) {
      return <p>No results found.</p>;
    }
    if ('Activities' in results) {
      return (
        <div>
          <h2 style={{
            fontFamily: 'Arial, sans-serif', marginTop: '5px', marginBottom: '0px' }}>
            {results['Compound Name']}
          </h2>
          <p style={{
            fontFamily: 'Arial, sans-serif', marginTop: '5px', marginBottom: '10px' }}>
            CID: {results.CID}
          </p>
          {imageUrl && <img src={imageUrl} alt="Compound Structure" />}
          
          <div className="classifications-container">
          <p style={{ color: '#4CAF50', fontSize: '20px' }}> PubChem Pharmacological Classifications:</p>
          <h2>{
            Array.isArray(results.Activities) && results.Activities.length > 0
              ? `[${results.Activities.join(' - ')}]`
              : 'No activities listed'
            }</h2>
        </div>
        </div>
      );
    } 
    else if ('Predicted Activities' in results) {
      return (
        <div>
          <h2 style={{
            fontFamily: 'Arial, sans-serif', marginTop: '5px' }}>
            {results['Compound Name']}
          </h2>
          <p style={{
            fontFamily: 'Arial, sans-serif', marginBottom: '0px' }}>
            CID: {results.CID}
          </p>
          {imageUrl && <img src={imageUrl} alt="Compound Structure" />}

          <div className="classifications-container">
          <p style={{ color: '#47a3ce'}}> Classification Unknown on PubChem </p>
          <p style={{ color: '#47a3ce', fontSize: '20px' }}> Predicted Pharmacological Classifications:</p>
            <h2>{
            Array.isArray(results['Predicted Activities']) && results['Predicted Activities'].length > 0
              ? `[${results['Predicted Activities'].join(' - ')}]`
              : 'Unable to Predict'
            }</h2>
          </div>
        </div>
      );
    }

    else if (Array.isArray(results)) {
      return (
        <div>
        <h2>Results for "{searchTerm}":</h2>
        <table style={{ borderRadius: '22px', overflow: 'hidden' }}>
          <thead>
            <tr>
              <th style={{ textAlign: 'center' }}>
                CID
              </th>
              <th style={{ textAlign: 'center', fontSize: '18px' }}>
                Compound Name
              </th>
            </tr>
          </thead>
          <tbody>
          {results.map((item, index) => (
              <tr key={index}
                  onMouseEnter={() => setHoveredIndex(index)}
                  onMouseLeave={() => setHoveredIndex(null)}>
                  <td onClick={() => handleCIDClick(item.CID)} 
                      style={{ 
                        cursor: 'pointer', 
                        color: hoveredIndex === index ? 'white' : '#45a049', 
                        textDecoration: 'underline',
                        textAlign: 'center',
                        fontFamily: 'Arial, sans-serif',
                        background: hoveredIndex === index ? '#45a049' : '#575757' 
                      }}>
                      {item.CID}
                  </td>
                  <td style={{ 
                        fontFamily: 'Arial, sans-serif',
                        fontSize: '15px',
                        background: hoveredIndex === index ? '#45a049' : '#575757' 
                      }}>
                    {item['Compound Name']}
                  </td>
              </tr>
          ))}
          </tbody>
        </table>
      </div>
      );
    } 
    else {
      return <p>No matching results.</p>;
    }
  };

    return (
            <div className="container">
                <h1>🧪 ℃ompound ⌬ Classifie℞ 💊</h1>
                <div className="search-bar">
                    <input
                        type="text"
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        placeholder="Enter Compound Name, CID, or Activity"
                    />
                    <button onClick={() => handleSearch()} style={{ marginRight: '10px' }}>Search</button>
                    <button onClick={fetchActivities}>Browse</button>
                </div>
                {error && <p className="error">{error}</p>}
                {activities.length > 0 && renderActivities()}
                {results && (
                    <div className="results-container">
                        {renderResults()}
                    </div>
                )}
            </div>
    );
};

export default App;

