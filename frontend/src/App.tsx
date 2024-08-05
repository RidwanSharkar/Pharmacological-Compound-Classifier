import React, { useState, useEffect } from 'react';
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
  'Predicted Activities': string[];
}

const App: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [results, setResults] = useState<Compound[] | ActivityResult | PredictionResult | null>(null);
  const [activities, setActivities] = useState<{ activity: string; count: number }[]>([]);
  const [error, setError] = useState('');
  const [isBrowsing, setIsBrowsing] = useState(false);
  
  const handleSearch = async () => {
    if (!searchTerm.trim()) {
      setError('Please enter a search term.');
      return;
    }
    setIsBrowsing(false);
    try {
      const response = await axios.post(`${FLASK_SERVER_URL}/search`, { searchTerm });
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
  };

  // hmm
  useEffect(() => {
    fetchActivities();
  }, []);

  const handleActivityClick = (activity: string) => {
    setSearchTerm(activity);
    handleSearch();
  };

  const renderActivities = () => {
    if (!activities.length || !isBrowsing) {
      return null;
    }
  
    return (
      <div>
        <h2>Available Activities:</h2>
        <table>
          <thead>
            <tr>
              <th>Activity</th>
              <th>Count</th>  {/* count COLUMN */}
            </tr>
          </thead>
          <tbody>
            {activities.map((item, index) => (
              <tr key={index}>
                <td>
                  <button onClick={() => handleActivityClick(item.activity)} style={{ textTransform: 'capitalize' }}>
                    {item.activity}
                  </button>
                </td>
                <td>{item.count}</td>  {/* count */}
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
          <h2>Details for CID "{results.CID}":</h2>
          <p>Compound Name: {results['Compound Name']}</p>
          <p>Activities: {Array.isArray(results.Activities) ? results.Activities.join(', ') : 'No activities listed'}</p>
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
                  <td>{item.CID}</td>
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
    <div className="container">
      <h1>Compound Activity Search</h1>
      <div className="search-bar">
        <input
          type="text"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          placeholder="Enter activity or CID"
        />
        <button onClick={handleSearch}>Search</button>
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

