import React, { useState } from 'react';
import axios from 'axios';
import './App.css';
import { BrowserRouter as Router, Route, Switch, Redirect } from 'react-router-dom';


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
  'Predicted Activities': string[];
}

interface Activity {
  activity: string;
  count: number;
}

interface SortConfig {
  key: keyof Activity;  // 'activity' or 'count'
  direction: 'ascending' | 'descending';
}

/*========================================================================================================*/


const App: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [results, setResults] = useState<Compound[] | ActivityResult | PredictionResult | null>(null);
  const [error, setError] = useState('');
  const [isBrowsing, setIsBrowsing] = useState(false);  // Swap Table <-> Search View
  const [sortConfig, setSortConfig] = useState<SortConfig>({ key: 'count', direction: 'ascending' }); // Sort
  const [activities, setActivities] = useState<Activity[]>([]);

  const [imageUrl, setImageUrl] = useState('');
  
  const handleSearch = async (cid?: string) => {
    const searchId = cid || searchTerm; // Use the provided CID if available, otherwise use the current searchTerm state

    if (!searchId.trim()) {
        setError('Please enter a search term.');
        return;
    }

    try {
        const detailResponse = await axios.post(`${FLASK_SERVER_URL}/search`, { searchTerm: searchId });
        if (detailResponse.data.error) {
            setError(detailResponse.data.error);
            setResults(null);
        } else {
            setResults(detailResponse.data);
            setError('');

            // Fetch the image URL using the CID from searchTerm
            const imageUrl = `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/CID/${searchId}/PNG`;
            setImageUrl(imageUrl); // Set the image URL directly without additional request
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

  /*========================================================================================================*/

  const renderActivities = () => {
    if (!activities.length || !isBrowsing) {
      return null;
    }
  
    const sortedActivities = [...activities].sort(sortActivities);

    return (
      <div>
        <h2>Available Activities:</h2>
        <table>
          <thead>
            <tr>
              <th>
                <button onClick={() => requestSort('activity')} style={{ border: 'none', background: 'none' }}>
                  Activity
                </button>
              </th>
              <th>
                <button onClick={() => requestSort('count')} style={{ border: 'none', background: 'none' }}>
                  Count
                </button>
              </th>
            </tr>
          </thead>
          <tbody>
            {sortedActivities.map((item, index) => (
              <tr key={index}>
                <td>
                  <button onClick={() => handleActivityClick(item.activity)} style={{ textTransform: 'capitalize' }}>
                    {item.activity}
                  </button>
                </td>
                <td>{item.count}</td>  
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };




  const renderResults = () => {
    if (!results) {
      return <p>No results found.</p>;
    }
    if ('Activities' in results) {
      return (
        <div>
          <h2>CID: {results.CID}</h2>
          <p>Compound Name:   {results['Compound Name']}</p>
          <p>Activities:   {Array.isArray(results.Activities) ? results.Activities.join(', ') : 'No activities listed'}</p>
          {imageUrl && <img src={imageUrl} alt="Compound Structure" />}
        </div>
      );
    } 
    else if (Array.isArray(results)) {
      return (
        <div>
          <h2>Results for "{searchTerm}":</h2>
          <table>
            <thead>
              <tr>
                <th>CID</th>
                <th>Compound Name</th>
              </tr>
            </thead>
            <tbody>
            {results.map((item, index) => (
                <tr key={index}>
                    <td onClick={() => handleCIDClick(item.CID)} style={{ cursor: 'pointer', color: 'blue', textDecoration: 'underline' }}>
                        {item.CID}
                    </td>
                    <td>{item['Compound Name']}</td>
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
        <Router>
            <div className="container">
                <h1>Compound Activity Search</h1>
                <div className="search-bar">
                    <input
                        type="text"
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        placeholder="Enter activity or CID"
                    />
                    <button onClick={() => handleSearch()}>Search</button>
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
        </Router>
    );
};

export default App;

