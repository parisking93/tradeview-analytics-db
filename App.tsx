
import { Database, LayoutDashboard, LineChart, Loader2, Search, Settings, Wallet, X } from 'lucide-react';
import React, { useEffect, useRef, useState } from 'react';
import ChartComponent from './components/ChartComponent';
import ForecastDashboard from './components/ForecastDashboard';
import PortfolioModal from './components/PortfolioModal';
import Sidebar from './components/Sidebar';
import TimeframeSelector from './components/TimeframeSelector';
import { fetchData, searchPairs } from './services/dataService';
import { Candle, PivotLevel, Trade } from './types';

const App: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [symbol, setSymbol] = useState('BTCUSD');
  const [timeframe, setTimeframe] = useState('1D'); // NEW State
  const [viewMode, setViewMode] = useState<'CHART' | 'FORECAST'>('CHART');

  const [candles, setCandles] = useState<Candle[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [pivots, setPivots] = useState<PivotLevel[]>([]);

  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<string[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const searchRef = useRef<HTMLDivElement>(null);

  const [isPortfolioOpen, setIsPortfolioOpen] = useState(false);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (searchRef.current && !searchRef.current.contains(event.target as Node)) {
        setShowResults(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  useEffect(() => {
    if (viewMode === 'FORECAST') return;

    const loadData = async () => {
      setLoading(true);
      try {
        const data = await fetchData(symbol, timeframe); // Pass timeframe
        setCandles(data.candles);
        setTrades(data.trades);
        setPivots(data.pivots);
      } catch (error) {
        console.error("Failed to fetch data:", error);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [symbol, viewMode, timeframe]); // Reload when timeframe changes

  const handleSearch = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const query = e.target.value;
    setSearchQuery(query);

    if (query.length > 1) {
      setIsSearching(true);
      setShowResults(true);
      try {
        const results = await searchPairs(query);
        setSearchResults(results);
      } catch (err) {
        console.error(err);
      } finally {
        setIsSearching(false);
      }
    } else {
      setSearchResults([]);
      setShowResults(false);
    }
  };

  const handleSelectSymbol = (newSymbol: string) => {
    if (newSymbol !== symbol) {
      setSymbol(newSymbol);
    }
    setSearchQuery('');
    setShowResults(false);
    setViewMode('CHART');
  };

  return (
    <div className="flex flex-col h-screen bg-gray-950 text-white overflow-hidden font-sans">

      <PortfolioModal
        isOpen={isPortfolioOpen}
        onClose={() => setIsPortfolioOpen(false)}
        onSelectTrade={handleSelectSymbol}
      />

      {/* Header */}
      <header className="h-16 border-b border-gray-800 bg-gray-900 flex items-center justify-between px-4 shrink-0 gap-4">
        <div className="flex items-center gap-3 min-w-fit cursor-pointer" onClick={() => setViewMode('CHART')}>
          <div className="bg-blue-600 p-1.5 rounded-md">
            <Database className="w-5 h-5 text-white" />
          </div>
          <h1 className="font-bold text-lg tracking-tight hidden sm:block">TradeView <span className="text-blue-500">DB</span></h1>
        </div>

        {/* Search Bar */}
        <div className={`flex-1 max-w-xl relative ${viewMode === 'FORECAST' ? 'opacity-50 pointer-events-none hidden md:block' : ''}`} ref={searchRef}>
          <div className="relative group">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <Search className="h-4 w-4 text-gray-500 group-focus-within:text-blue-400" />
            </div>
            <input
              type="text"
              className="block w-full pl-10 pr-3 py-2 border border-gray-700 rounded-lg leading-5 bg-gray-800 text-gray-300 placeholder-gray-500 focus:outline-none focus:bg-gray-900 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 sm:text-sm transition-all"
              placeholder="Search Pair (e.g. BTCUSD, EUR...)"
              value={searchQuery}
              onChange={handleSearch}
              onFocus={() => { if (searchQuery.length > 1) setShowResults(true); }}
            />
            {searchQuery && (
              <button
                onClick={() => { setSearchQuery(''); setShowResults(false); }}
                className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-500 hover:text-white"
              >
                <X className="h-4 w-4" />
              </button>
            )}
          </div>

          {showResults && (
            <div className="absolute mt-1 w-full bg-gray-800 border border-gray-700 rounded-md shadow-xl z-50 max-h-60 overflow-y-auto">
              {isSearching ? (
                <div className="p-4 text-center text-gray-400 text-sm">
                  <Loader2 className="w-4 h-4 animate-spin inline mr-2" />
                  Searching currency table...
                </div>
              ) : searchResults.length > 0 ? (
                <ul>
                  {searchResults.map((res) => (
                    <li
                      key={res}
                      onClick={() => handleSelectSymbol(res)}
                      className="px-4 py-2 hover:bg-gray-700 cursor-pointer flex justify-between items-center group"
                    >
                      <span className="font-medium text-gray-200">{res}</span>
                      <span className="text-xs text-gray-500 group-hover:text-blue-400">Select</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <div className="p-4 text-center text-gray-500 text-sm">
                  No pairs found.
                </div>
              )}
            </div>
          )}
        </div>

        <div className="flex items-center gap-2 sm:gap-4 min-w-fit">
          <button
            onClick={() => setViewMode(viewMode === 'CHART' ? 'FORECAST' : 'CHART')}
            className={`flex items-center gap-2 px-3 py-2 rounded transition-colors font-medium text-sm border ${viewMode === 'FORECAST'
              ? 'bg-purple-600 text-white border-purple-500'
              : 'bg-gray-800 text-gray-300 border-gray-700 hover:bg-gray-700'
              }`}
          >
            <LineChart className="w-4 h-4" />
            <span className="hidden sm:inline">AI Forecast</span>
          </button>

          <button
            onClick={() => setIsPortfolioOpen(true)}
            className="flex items-center gap-2 px-3 py-2 bg-gray-800 hover:bg-gray-700 text-white rounded transition-colors font-medium text-sm border border-gray-700"
          >
            <Wallet className="w-4 h-4" />
            <span className="hidden sm:inline">My Portfolio</span>
          </button>

          <div className="h-6 w-px bg-gray-800 mx-1"></div>

          <button className="p-2 hover:bg-gray-800 rounded text-gray-400">
            <LayoutDashboard className="w-5 h-5" />
          </button>
          <button className="p-2 hover:bg-gray-800 rounded text-gray-400">
            <Settings className="w-5 h-5" />
          </button>
        </div>
      </header>

      {/* Main Content Area */}
      <div className="flex flex-1 overflow-hidden">

        {viewMode === 'FORECAST' ? (
          <div className="w-full h-full">
            <ForecastDashboard />
          </div>
        ) : (
          <>
            <main className="flex-1 relative flex flex-col">
              {/* Timeframe Bar (In Main View) */}
              <div className="h-10 border-b border-gray-800 bg-gray-900/50 flex items-center px-4 justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold text-gray-400 uppercase tracking-wider mr-2">Interval</span>
                  <TimeframeSelector selected={timeframe} onSelect={setTimeframe} />
                </div>
              </div>

              {loading ? (
                <div className="flex-1 flex items-center justify-center flex-col gap-4">
                  <Loader2 className="w-10 h-10 text-blue-500 animate-spin" />
                  <p className="text-gray-500 animate-pulse">Fetching {symbol} ({timeframe}) data...</p>
                </div>
              ) : (
                <div className="w-full h-full">
                  <ChartComponent
                    data={candles}
                    trades={trades}
                    pivots={pivots}
                    symbol={symbol}
                    timeframe={timeframe}
                  />
                </div>
              )}
            </main>

            <aside className="hidden md:block h-full shadow-2xl z-10">
              <Sidebar trades={trades} onSelectTrade={handleSelectSymbol} />
            </aside>
          </>
        )}

      </div>

      {viewMode === 'CHART' && (
        <div className="md:hidden h-1/3 border-t border-gray-800 bg-gray-900 overflow-hidden">
          <Sidebar trades={trades} onSelectTrade={handleSelectSymbol} />
        </div>
      )}
    </div>
  );
};

export default App;
