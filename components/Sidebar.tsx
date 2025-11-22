import React from 'react';
import { Trade, TradeType } from '../types';
import { ArrowUpCircle, ArrowDownCircle, TrendingUp, AlertCircle, MousePointerClick } from 'lucide-react';

interface SidebarProps {
  trades: Trade[];
  onSelectTrade: (symbol: string) => void;
}

const Sidebar: React.FC<SidebarProps> = ({ trades, onSelectTrade }) => {
  return (
    <div className="w-full md:w-96 bg-gray-950 border-l border-gray-800 h-full flex flex-col overflow-hidden">
      <div className="p-4 border-b border-gray-800">
        <h3 className="text-lg font-semibold text-white flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-blue-400" />
          Trade Journal
        </h3>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {trades.length === 0 && (
            <div className="text-gray-500 text-center mt-10">No trades found.</div>
        )}
        {trades.map((trade) => (
          <div 
            key={trade.id} 
            onClick={() => onSelectTrade(trade.symbol)}
            className={`p-4 rounded-lg border transition-all duration-200 cursor-pointer group hover:scale-[1.02] ${
              trade.status === 'OPEN' 
                ? 'border-blue-500/30 bg-blue-500/5 hover:border-blue-500/60' 
                : 'border-gray-800 bg-gray-900 hover:border-gray-600'
            }`}
          >
            <div className="flex justify-between items-start mb-3">
              <div className="flex items-center gap-2">
                {trade.type === TradeType.LONG ? (
                  <ArrowUpCircle className="w-5 h-5 text-trade-up" />
                ) : (
                  <ArrowDownCircle className="w-5 h-5 text-trade-down" />
                )}
                <span className="font-bold text-white group-hover:text-blue-400 transition-colors">
                    {trade.symbol}
                </span>
              </div>
              <span className={`text-xs px-2 py-1 rounded font-medium ${
                trade.status === 'OPEN' ? 'bg-blue-500/20 text-blue-400' : 'bg-gray-700 text-gray-300'
              }`}>
                {trade.status}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="flex flex-col">
                <span className="text-gray-500 text-xs">Entry</span>
                <span className="text-gray-200 font-mono">{trade.entryPrice}</span>
              </div>
              <div className="flex flex-col text-right">
                 <span className="text-gray-500 text-xs">Date</span>
                 <span className="text-gray-200 text-xs">{trade.entryDate}</span>
              </div>
              
              <div className="flex flex-col mt-2">
                 <span className="text-gray-500 text-xs">Stop Loss</span>
                 <span className={`font-mono ${trade.stopLoss === 0 ? 'text-gray-600' : 'text-red-400'}`}>
                    {trade.stopLoss || '-'}
                 </span>
              </div>
              <div className="flex flex-col mt-2 text-right">
                 <span className="text-gray-500 text-xs">Take Profit</span>
                 <span className={`font-mono ${trade.takeProfit === 0 ? 'text-gray-600' : 'text-green-400'}`}>
                    {trade.takeProfit || '-'}
                 </span>
              </div>
            </div>

            {trade.status === 'CLOSED' && trade.pnl !== undefined && (
                <div className={`mt-3 pt-2 border-t border-gray-800 flex justify-between items-center ${trade.pnl > 0 ? 'text-trade-up' : 'text-trade-down'}`}>
                    <span className="text-xs font-semibold">Realized P&L</span>
                    <span className="font-mono font-bold">
                        {trade.pnl > 0 ? '+' : ''}{trade.pnl} USD
                    </span>
                </div>
            )}
            
            <div className="mt-2 flex items-center justify-end opacity-0 group-hover:opacity-100 transition-opacity">
                <span className="text-[10px] text-gray-500 flex items-center gap-1">
                    View Chart <MousePointerClick className="w-3 h-3" />
                </span>
            </div>
          </div>
        ))}
      </div>

      <div className="p-4 border-t border-gray-800 bg-gray-900">
        <div className="flex items-start gap-2 text-yellow-500/80 text-xs">
            <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
            <p>
                Click on an order to switch the chart to that symbol.
            </p>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;