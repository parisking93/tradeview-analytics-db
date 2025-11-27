import React from 'react';

interface TimeframeSelectorProps {
  selected: string;
  onSelect: (tf: string) => void;
}

const TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1D'];

const TimeframeSelector: React.FC<TimeframeSelectorProps> = ({ selected, onSelect }) => {
  return (
    <div className="flex items-center gap-1 bg-gray-900 p-1 rounded-md border border-gray-800">
      {TIMEFRAMES.map((tf) => (
        <button
          key={tf}
          onClick={() => onSelect(tf)}
          className={`px-3 py-1 text-xs font-medium rounded transition-all ${
            selected === tf
              ? 'bg-gray-700 text-white shadow-sm'
              : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'
          }`}
        >
          {tf}
        </button>
      ))}
    </div>
  );
};

export default TimeframeSelector;