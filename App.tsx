// --- START OF FILE App.tsx ---

import { Database, Loader2, Wallet } from 'lucide-react';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import ChartComponent from './components/ChartComponent';
import ForecastDashboard from './components/ForecastDashboard';
import PortfolioModal from './components/PortfolioModal';
import Sidebar from './components/Sidebar';
import TimeframeSelector from './components/TimeframeSelector';
import { fetchData, searchPairs } from './services/dataService';
// Assicurati di aggiornare dataService per supportare i parametri di data!
import { LayoutDashboard, LineChart, Search, Settings, X } from 'lucide-react';
import { Candle, PivotLevel, Trade } from './types';
const App: React.FC = () => {
  const [initialLoading, setInitialLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);

  const [symbol, setSymbol] = useState('ETHEUR');
  const [timeframe, setTimeframe] = useState('1h');
  const [viewMode, setViewMode] = useState<'CHART' | 'FORECAST'>('CHART');

  const [candles, setCandles] = useState<Candle[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [pivots, setPivots] = useState<PivotLevel[]>([]);

  // Search Logic (invariata)
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<string[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [isPortfolioOpen, setIsPortfolioOpen] = useState(false);
  const searchRef = useRef<HTMLDivElement>(null);

  // 1. Caricamento Iniziale (Reset Totale)
  useEffect(() => {
    if (viewMode === 'FORECAST') return;

    const loadInitialData = async () => {
      setInitialLoading(true);
      setCandles([]); // Reset candele al cambio simbolo/timeframe
      try {
        // fetch iniziale senza endDate -> prende le ultime N candele (es. ultimi 5 giorni)
        const data = await fetchData(symbol, timeframe, undefined);
        setCandles(data.candles);
        setTrades(data.trades);
        setPivots(data.pivots);
      } catch (error) {
        console.error("Failed to fetch initial data:", error);
      } finally {
        setInitialLoading(false);
      }
    };

    loadInitialData();
  }, [symbol, viewMode, timeframe]);

  // 2. Funzione per caricare lo storico (Infinite Scroll)
  const handleLoadMore = useCallback(async (earliestTimestamp: number) => {
    if (loadingMore || initialLoading) return;

    setLoadingMore(true);
    try {
      // Convertiamo il timestamp (ms) in formato data stringa per il DB
      // earliestTimestamp è l'inizio del grafico attuale. Vogliamo dati PRIMA di questo.
      const dateObj = new Date(earliestTimestamp);
      const endDateString = dateObj.toISOString().slice(0, 19).replace('T', ' '); // '2025-11-27 08:00:00'

      console.log(`Caricamento storico per ${symbol} prima del: ${endDateString}`);

      // Chiamata API passando endDateString come limite superiore ("to")
      // Il backend userà la query: WHERE timestamp < endDateString ORDER BY timestamp DESC LIMIT 500
      const newHistoryData = await fetchData(symbol, timeframe, endDateString);

      if (newHistoryData.candles.length > 0) {
        setCandles(prevCandles => {
          // Unione: Nuove candele (vecchie) + Candele Esistenti (nuove)
          // Assicuriamoci che non ci siano duplicati basati sul tempo
          const newCandlesFiltered = newHistoryData.candles.filter(
            nc => !prevCandles.some(pc => pc.time === nc.time)
          );
          return [...newCandlesFiltered, ...prevCandles].sort((a, b) => new Date(a.time).getTime() - new Date(b.time).getTime());
        });
      }
    } catch (err) {
      console.error("Error loading history:", err);
    } finally {
      setLoadingMore(false);
    }
  }, [symbol, timeframe, loadingMore, initialLoading]);

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
    if (newSymbol !== symbol) setSymbol(newSymbol);
    setShowResults(false);
    setViewMode('CHART');
  };

  return (
    <div className="flex flex-col h-screen bg-gray-950 text-white overflow-hidden font-sans">

      <PortfolioModal isOpen={isPortfolioOpen} onClose={() => setIsPortfolioOpen(false)} onSelectTrade={handleSelectSymbol} />

      {/* Header omesso per brevità, identico a prima ... */}
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
          <TimeframeSelector selected={timeframe} onSelect={setTimeframe} />

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
      <div className="flex flex-1 overflow-hidden min-w-0">
        {viewMode === 'FORECAST' ? (
          <ForecastDashboard />
        ) : (
          <main className="flex-1 min-w-0 relative flex flex-col overflow-hidden">
            {initialLoading ? (
              <div className="flex-1 flex items-center justify-center flex-col gap-4">
                <Loader2 className="w-10 h-10 text-blue-500 animate-spin" />
                <p className="text-gray-500">Loading {symbol}...</p>
              </div>
            ) : (
              <div className="w-full h-full relative">
                <ChartComponent
                  data={candles}
                  trades={trades}
                  pivots={pivots}
                  symbol={symbol}
                  timeframe={timeframe}
                  onRequestMore={handleLoadMore} // Passiamo la funzione
                  isLoadingMore={loadingMore}    // Passiamo lo stato
                />
              </div>
            )}
          </main>
        )}
        <aside className="hidden md:block w-80 h-full border-l border-gray-800">
          <Sidebar trades={trades} onSelectTrade={handleSelectSymbol} />
        </aside>
      </div>
    </div>
  );
};

export default App;
