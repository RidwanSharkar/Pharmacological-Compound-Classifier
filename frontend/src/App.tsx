import React, { useState } from 'react';

const App: React.FC = () => {
  const [cid, setCid] = useState('');
  const [result, setResult] = useState('');
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
        setResult(JSON.stringify(data, null, 2));
        setError('');
      } else {
        throw new Error(data.error || 'Error fetching data');
      }
    } catch (error) {
      setError(`Failed to fetch: ${error.message}`);
      setResult('');
    }
  };

  return (
    <div>
      <h1>Pharmacological & Chemical Compound Classifier</h1>
      <input
        type="text"
        value={cid}
        onChange={(e) => setCid(e.target.value)}
        placeholder="Enter CID"
      />
      <button onClick={handleSearch}>Search</button>
      {result && <pre>{result}</pre>}
      {error && <p style={{ color: 'red' }}>{error}</p>}
    </div>
  );
};

export default App;
