
import React, { useEffect, useState } from 'react';
import { Trade, Candle } from '../types';
import { fetchData, fetchPortfolio } from '../services/dataService';
import ForecastChart from './ForecastChart';
import TimeframeSelector from './TimeframeSelector';
import { ArrowUpCircle, ArrowDownCircle, Loader2, BrainCircuit } from 'lucide-react';

const ForecastDashboard: React.FC = () => {
    const [portfolioTrades, setPortfolioTrades] = useState<Trade[]>([]);
    const [selectedTrade, setSelectedTrade] = useState<Trade | null>(null);
    const [timeframe, setTimeframe] = useState('1h'); // Default better for forecast
    const [chartData, setChartData] = useState<{ candles: Candle[], forecast: Candle[] }>({ candles: [], forecast: [] });
    const [displayMode, setDisplayMode] = useState<'candles' | 'line'>('candles');
    const [showForecastBand, setShowForecastBand] = useState(true);
    const [loading, setLoading] = useState(true);
    const [loadingChart, setLoadingChart] = useState(false);

    // 1. Load Portfolio on Mount
    useEffect(() => {
        const loadPortfolio = async () => {
            try {
                const trades = await fetchPortfolio();
                setPortfolioTrades(trades);
                if (trades.length > 0) {
                    setSelectedTrade(trades[0]);
                } else {
                    setLoading(false);
                }
            } catch (e) {
                console.error(e);
                setLoading(false);
            }
        };
        loadPortfolio();
    }, []);

    // 2. Load Chart Data when Selected Trade OR Timeframe Changes
    useEffect(() => {
        if (!selectedTrade) return;

        const loadChartData = async () => {
            setLoadingChart(true);
            try {
                const data = await fetchData(selectedTrade.symbol, timeframe);
                setChartData({
                    candles: data.candles,
                    forecast: data.forecast || []
                });
            } catch (e) {
                console.error(e);
            } finally {
                setLoading(false);
                setLoadingChart(false);
            }
        };

        loadChartData();
    }, [selectedTrade, timeframe]);

    if (loading) {
        return <div className="flex items-center justify-center h-full text-white"><Loader2 className="animate-spin mr-2"/> Loading Dashboard...</div>;
    }

    return (
        <div className="flex h-full bg-gray-950 min-w-0 flex-1 w-full">
            
            {/* Main Forecast Chart Area */}
            <div className="flex-1 min-w-0 flex flex-col border-r border-gray-800 relative">
                
                {/* Forecast Toolbar */}
                <div className="h-10 border-b border-gray-800 bg-gray-900/50 flex items-center px-4 justify-between">
                     <div className="flex items-center gap-2">
                        <span className="text-xs font-bold text-gray-400 uppercase tracking-wider mr-2">Interval</span>
                        <TimeframeSelector selected={timeframe} onSelect={setTimeframe} />
                     </div>
                     <div className="flex items-center gap-2">
                        <button
                          onClick={() => {
                            const next = displayMode === 'candles' ? 'line' : 'candles';
                            setDisplayMode(next);
                            if (next === 'candles') setShowForecastBand(true);
                          }}
                          className="px-3 py-1.5 rounded-md border border-gray-700 bg-gray-800 hover:bg-gray-700 text-xs font-semibold text-gray-200 transition-colors"
                        >
                          Switch Mode: {displayMode === 'candles' ? 'Candles' : 'Line'}
                        </button>
                        {displayMode === 'line' && (
                          <button
                            onClick={() => setShowForecastBand(v => !v)}
                            className={`px-3 py-1.5 rounded-md border text-xs font-semibold transition-colors ${showForecastBand
                              ? 'border-green-700 bg-green-900/40 text-green-200 hover:bg-green-900/60'
                              : 'border-gray-700 bg-gray-800 text-gray-200 hover:bg-gray-700'}`}
                          >
                            {showForecastBand ? 'Hide Band' : 'Show Band'}
                          </button>
                        )}
                     </div>
                </div>

                <div className="flex-1 w-full relative min-w-0">
                    {selectedTrade ? (
                        loadingChart ? (
                            <div className="absolute inset-0 flex items-center justify-center bg-gray-950/50 backdrop-blur-sm z-20">
                                <Loader2 className="w-10 h-10 text-blue-500 animate-spin" />
                            </div>
                        ) : (
                            <ForecastChart 
                                candles={chartData.candles} 
                                forecast={chartData.forecast} 
                                activeTrade={selectedTrade}
                                symbol={selectedTrade.symbol}
                                timeframe={timeframe}
                                displayMode={displayMode}
                                showBand={showForecastBand}
                            />
                        )
                    ) : (
                        <div className="flex items-center justify-center h-full text-gray-500">
                            Select a position to view forecast.
                        </div>
                    )}
                </div>
            </div>

            {/* Right Sidebar: Active Positions */}
            <div className="w-80 bg-gray-900 flex flex-col border-l border-gray-800 shadow-xl z-10">
                <div className="p-4 border-b border-gray-800 bg-gray-900">
                    <h2 className="text-white font-semibold flex items-center gap-2">
                        <BrainCircuit className="w-5 h-5 text-purple-500" />
                        AI Forecast Monitor
                    </h2>
                    <p className="text-xs text-gray-500 mt-1">Select a position to analyze prediction accuracy.</p>
                </div>
                
                <div className="flex-1 overflow-y-auto p-2 space-y-2">
                    {portfolioTrades.length === 0 && (
                        <div className="text-center text-gray-500 mt-10 text-sm">
                            No open positions found.
                        </div>
                    )}
                    
                    {portfolioTrades.map(trade => (
                        <div 
                            key={trade.id}
                            onClick={() => setSelectedTrade(trade)}
                            className={`p-3 rounded-lg cursor-pointer border transition-all ${
                                selectedTrade?.id === trade.id 
                                ? 'bg-gray-800 border-purple-500 shadow-lg shadow-purple-900/20' 
                                : 'bg-gray-800/30 border-transparent hover:bg-gray-800 hover:border-gray-700'
                            }`}
                        >
                            <div className="flex justify-between items-start mb-2">
                                <span className="font-bold text-gray-200 text-sm">{trade.symbol}</span>
                                {trade.type === 'LONG' ? <ArrowUpCircle className="w-4 h-4 text-green-500"/> : <ArrowDownCircle className="w-4 h-4 text-red-500"/>}
                            </div>
                            
                            <div className="flex justify-between items-end text-xs">
                                <div className="flex flex-col text-gray-500">
                                    <span>Entry</span>
                                    <span className="font-mono text-gray-300">{trade.entryPrice}</span>
                                </div>
                                <div className="flex flex-col text-gray-500 text-right">
                                    <span>P&L (Est)</span>
                                    <span className="font-mono text-blue-400">Live</span>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
                
                {/* Mini Portfolio Stats */}
                <div className="p-4 bg-gray-950 border-t border-gray-800">
                    <div className="text-xs text-gray-400 uppercase tracking-wider mb-2">Total Equity Exposure</div>
                    <div className="text-xl font-mono text-white font-bold">$12,450.00</div>
                </div>
            </div>
        </div>
    );
};

export default ForecastDashboard;
