/* frontend\src\App.tsx: */

import React, { useState } from 'react';
import './App.css'; 

const App: React.FC = () => {
  const [cid, setCid] = useState('');
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState('');

  const handleSearch = async () => {
    if (!cid.trim()) {
      alert('Please enter a CID.');
      return;
    }

    const requestOptions = {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ cid })
    };

    try {
      const response = await fetch('http://127.0.0.1:5000/predict', requestOptions);
      const data = await response.json();
      if (response.ok) {
        setResult(data);
        setError('');
      } else {
        throw new Error(data.error || 'Error fetching data');
      }
    } catch (error: unknown) {
      if (error instanceof Error) {
        setError(`Failed to fetch: ${error.message}`);
      } else {
        setError('Failed to fetch: An unexpected error occurred');
      }
      setResult(null);
    }
  };

  return (
    <div className="container">
      <h1>Pharmacological & Chemical Compound Classifier</h1>
      <div className="search-bar">
        <input
          type="text"
          value={cid}
          onChange={(e) => setCid(e.target.value)}
          placeholder="Enter CID"
        />
        <button onClick={handleSearch}>Search</button>
      </div>
      <div className="results-container">
        {result && (
          <>
            <div className="result-box">
              <h3>Chemical Structure</h3>
              <img src={result.image} alt="Chemical Structure" />
            </div>
            <div className="result-box">
              <h3>Molecule Data</h3>
              <p>{result.data}</p>
            </div>
            <div className="result-box">
              <h3>Predicted Action</h3>
              <p>{result.action}</p>
            </div>
          </>
        )}
        {error && <p style={{ color: 'red' }}>{error}</p>}
      </div>
    </div>
  );
};

export default App;