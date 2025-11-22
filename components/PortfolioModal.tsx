import React, { useEffect, useState } from 'react';
import { Trade, TradeType } from '../types';
import { fetchPortfolio } from '../services/dataService';
import { X, Wallet, ArrowUpCircle, ArrowDownCircle, Loader2, ExternalLink } from 'lucide-react';

interface PortfolioModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectTrade: (symbol: string) => void;
}

const PortfolioModal: React.FC<PortfolioModalProps> = ({ isOpen, onClose, onSelectTrade }) => {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (isOpen) {
      setLoading(true);
      fetchPortfolio()
        .then(data => setTrades(data))
        .catch(err => console.error(err))
        .finally(() => setLoading(false));
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
      <div className="bg-gray-900 border border-gray-700 w-full max-w-2xl rounded-xl shadow-2xl flex flex-col max-h-[80vh]">
        
        {/* Header */}
        <div className="flex justify-between items-center p-5 border-b border-gray-800">
          <div className="flex items-center gap-3">
            <div className="bg-purple-600 p-2 rounded-lg">
              <Wallet className="w-5 h-5 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white">My Portfolio</h2>
              <p className="text-xs text-gray-400">Active & Pending Orders</p>
            </div>
          </div>
          <button onClick={onClose} className="text-gray-400 hover:text-white p-2 rounded-full hover:bg-gray-800 transition">
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-5">
          {loading ? (
            <div className="flex flex-col items-center justify-center h-40 gap-3 text-gray-500">
              <Loader2 className="w-8 h-8 animate-spin text-purple-500" />
              <p>Loading positions...</p>
            </div>
          ) : trades.length === 0 ? (
            <div className="text-center text-gray-500 py-10">
              No open positions found.
            </div>
          ) : (
            <div className="grid gap-4">
              {trades.map((trade) => (
                <div 
                  key={trade.id}
                  onClick={() => {
                    onSelectTrade(trade.symbol);
                    onClose();
                  }}
                  className="group bg-gray-800/50 border border-gray-700 hover:border-purple-500/50 hover:bg-gray-800 rounded-lg p-4 cursor-pointer transition-all duration-200 relative overflow-hidden"
                >
                  {/* Hover Effect Gradient */}
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-transparent to-purple-500/5 opacity-0 group-hover:opacity-100 transition-opacity" />

                  <div className="flex justify-between items-start relative z-10">
                    
                    {/* Left Side: Symbol & Type */}
                    <div className="flex items-center gap-3">
                      {trade.type === TradeType.LONG ? (
                        <div className="bg-trade-up/10 p-2 rounded-full">
                           <ArrowUpCircle className="w-6 h-6 text-trade-up" />
                        </div>
                      ) : (
                        <div className="bg-trade-down/10 p-2 rounded-full">
                           <ArrowDownCircle className="w-6 h-6 text-trade-down" />
                        </div>
                      )}
                      <div>
                        <h3 className="font-bold text-lg text-white flex items-center gap-2">
                          {trade.symbol}
                          <span className={`text-[10px] px-1.5 py-0.5 rounded border ${
                            trade.status === 'OPEN' ? 'border-green-500/30 text-green-400' : 'border-yellow-500/30 text-yellow-400'
                          }`}>
                            {trade.status}
                          </span>
                        </h3>
                        <p className="text-sm text-gray-400 flex items-center gap-1">
                          Entry: <span className="font-mono text-gray-200">{trade.entryPrice}</span>
                        </p>
                      </div>
                    </div>

                    {/* Right Side: Stats & Action */}
                    <div className="text-right">
                        <div className="flex items-center justify-end gap-1 text-xs text-gray-500 mb-1 group-hover:text-purple-400 transition-colors">
                            Open Chart <ExternalLink className="w-3 h-3" />
                        </div>
                        <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
                            <span className="text-gray-500 text-xs">TP</span>
                            <span className="font-mono text-green-400">{trade.takeProfit}</span>
                            <span className="text-gray-500 text-xs">SL</span>
                            <span className="font-mono text-red-400">{trade.stopLoss}</span>
                        </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-gray-800 bg-gray-900/50 text-center text-xs text-gray-500">
          Total Open Positions: <span className="text-white font-bold">{trades.length}</span>
        </div>
      </div>
    </div>
  );
};

export default PortfolioModal;