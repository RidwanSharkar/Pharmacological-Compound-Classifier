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
  'Predicted Activities': string[];
}

const App: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [results, setResults] = useState<Compound[] | ActivityResult | PredictionResult | null>(null);
  const [error, setError] = useState('');

  const handleSearch = async () => {
    if (!searchTerm.trim()) {
      setError('Please enter a search term.');
      return;
    }
  
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

const renderResults = () => {
  if (!results) {
      return <p>No results found.</p>;
  }

  if (Array.isArray(results)) {
      return (
          <ul>
              {results.map((compound) => (
                  <li key={compound.CID}>
                      CID: {compound.CID} - {compound['Compound Name']}
                  </li>
              ))}
          </ul>
      );
  } else if ('Activities' in results) {
      return (
          <div>
              <p>CID: {results.CID}</p>
              <p>Compound Name: {results['Compound Name']}</p>
              <p>Activities: {Array.isArray(results.Activities) ? results.Activities.join(', ') : results.Activities}</p>
          </div>
      );
  } else {
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
      </div>
      {error && <p className="error">{error}</p>}
      {results && (
        <div className="results-container">
          <h2>Results for "{searchTerm}":</h2>
          {renderResults()}
        </div>
      )}
    </div>
  );
};

export default App;
