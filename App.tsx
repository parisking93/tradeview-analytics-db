import React, { useEffect, useState, useRef } from 'react';
import ChartComponent from './components/ChartComponent';
import Sidebar from './components/Sidebar';
import PortfolioModal from './components/PortfolioModal';
import { fetchData, searchPairs } from './services/dataService';
import { Candle, Trade, PivotLevel } from './types';
import { Database, LayoutDashboard, Settings, Loader2, Search, X, Wallet } from 'lucide-react';

const App: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [symbol, setSymbol] = useState('BTCUSD');
  
  // Data State
  const [candles, setCandles] = useState<Candle[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [pivots, setPivots] = useState<PivotLevel[]>([]);

  // Search State
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<string[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const searchRef = useRef<HTMLDivElement>(null);

  // Portfolio Modal State
  const [isPortfolioOpen, setIsPortfolioOpen] = useState(false);

  // Close search dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (searchRef.current && !searchRef.current.contains(event.target as Node)) {
        setShowResults(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Load Data when Symbol changes
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      try {
        const data = await fetchData(symbol);
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
  }, [symbol]);

  // Handle Search Input
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

  // Select Symbol from Search or Sidebar or Portfolio
  const handleSelectSymbol = (newSymbol: string) => {
    if (newSymbol === symbol) return;
    setSymbol(newSymbol);
    setSearchQuery('');
    setShowResults(false);
  };

  return (
    <div className="flex flex-col h-screen bg-gray-950 text-white overflow-hidden font-sans">
      
      {/* Portfolio Modal */}
      <PortfolioModal 
        isOpen={isPortfolioOpen} 
        onClose={() => setIsPortfolioOpen(false)} 
        onSelectTrade={handleSelectSymbol}
      />

      {/* Header */}
      <header className="h-16 border-b border-gray-800 bg-gray-900 flex items-center justify-between px-4 shrink-0 gap-4">
        <div className="flex items-center gap-3 min-w-fit">
          <div className="bg-blue-600 p-1.5 rounded-md">
            <Database className="w-5 h-5 text-white" />
          </div>
          <h1 className="font-bold text-lg tracking-tight hidden sm:block">TradeView <span className="text-blue-500">DB</span></h1>
        </div>

        {/* Search Bar */}
        <div className="flex-1 max-w-xl relative" ref={searchRef}>
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
                    onFocus={() => { if(searchQuery.length > 1) setShowResults(true); }}
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

            {/* Search Results Dropdown */}
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
                            No pairs found in database.
                        </div>
                    )}
                </div>
            )}
        </div>
        
        <div className="flex items-center gap-2 sm:gap-4 min-w-fit">
            <div className="hidden md:flex items-center gap-2 text-sm text-gray-400 bg-gray-800 px-3 py-1.5 rounded-full border border-gray-700">
                <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
                <span className="text-xs font-mono">Python Backend</span>
            </div>
            
            {/* NEW Portfolio Button */}
            <button 
                onClick={() => setIsPortfolioOpen(true)}
                className="flex items-center gap-2 px-3 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded transition-colors font-medium text-sm shadow-lg shadow-purple-900/20"
            >
                <Wallet className="w-4 h-4" />
                <span className="hidden sm:inline">Portfolio</span>
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

      {/* Main Content */}
      <div className="flex flex-1 overflow-hidden">
        
        {/* Chart Area */}
        <main className="flex-1 relative flex flex-col">
          {loading ? (
            <div className="flex-1 flex items-center justify-center flex-col gap-4">
                <Loader2 className="w-10 h-10 text-blue-500 animate-spin" />
                <p className="text-gray-500 animate-pulse">Fetching {symbol} data...</p>
            </div>
          ) : (
            <div className="w-full h-full">
                <ChartComponent 
                    data={candles} 
                    trades={trades} 
                    pivots={pivots}
                    symbol={symbol}
                />
            </div>
          )}
        </main>

        {/* Right Sidebar */}
        <aside className="hidden md:block h-full shadow-2xl z-10">
             <Sidebar trades={trades} onSelectTrade={handleSelectSymbol} />
        </aside>

      </div>
      
      {/* Mobile Trade List (Visible only on small screens) */}
      <div className="md:hidden h-1/3 border-t border-gray-800 bg-gray-900 overflow-hidden">
         <Sidebar trades={trades} onSelectTrade={handleSelectSymbol} />
      </div>
    </div>
  );
};

export default App;