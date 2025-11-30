from db.KrakenManager import KrakenPortfolioManager
from db.DatabaseManager import DatabaseManager
from db.MarketDataProvider import MarketDataProvider # Assicurati del percorso import corretto
from db.TimeSfmForecaster import TimeSfmForecaster
def main():
    print("=== AVVIO BOT ===")

    # 1. GESTIONE PORTAFOGLIO (KrakenManager)
    print("\n[1] Analisi Portafoglio...")
    kraken_mgr = KrakenPortfolioManager()
    kraken_mgr.get_open_orders()
    kraken_mgr.get_open_positions()
    kraken_mgr.get_normalized_portfolio()

    summary_data = kraken_mgr.get_portfolio_summary()
    all_positions = kraken_mgr.get_all_positions_data()

    # 2. GESTIONE DATI DI MERCATO (MarketDataProvider)
    print("\n[2] Analisi Mercato...")
    market_prov = MarketDataProvider()

    # Definiamo la coppia da analizzare
    TARGET_PAIR = "XBT/EUR"

    # Recupero Info Coppia
    pair_info = market_prov.getPair(TARGET_PAIR)

    # data_now = market_prov.getCandles(TARGET_PAIR, "now")


    # 3. SALVATAGGIO NEL DATABASE
    print("\n[3] Salvataggio Database...")
    db = DatabaseManager()

    # A. Salva Wallet e Ordini
    wallet_id = db.insert_wallet(summary_data)
    if wallet_id:
        db.insert_orders(wallet_id, all_positions)

    # B. Salva Dati Currency (Market Data)
    # Nota: Funziona solo se esiste la coppia in pair_limits nel DB
    # db.insert_currency_data(merged_candles, pair_info)

    db.close_connection()
    print("\n=== FINE ===")


def main2():
    market_prov = MarketDataProvider()
    # 1. Recupera TUTTE le coppie EUR con leva da Kraken
    all_pairs_eur = market_prov.getAllPairs(quote_filter="EUR", leverage_only=True)

    # 2. Salva le coppie nel DB (Operazione una tantum o di aggiornamento)
    db = DatabaseManager()
    db.insert_all_pairs(all_pairs_eur)
    db.close_connection()
    print("\n=== FINE ===")

def main3():
    market_prov = MarketDataProvider()
    db = DatabaseManager()
    all_pairs_eur = market_prov.getAllPairs(quote_filter="EUR", leverage_only=True)
    for p in all_pairs_eur:
        # currency = market_prov.getCandles(p['pair'], "15m", "1mo")
        # data_1d = market_prov.getCandles(p['pair'], "1d", "5d",)
        data_4h = market_prov.getCandles(p['pair'], "4h", "1mo",)
        data_1h = market_prov.getCandles(p['pair'], "1h", "1mo", )
        # data_15m = market_prov.getCandles(p['pair'], "15m", "1d")
        # data_5m = market_prov.getCandles(p['pair'], "5m", "1d")
        # currency = market_prov.getCandles(p['pair'], "1m", "1d")
        # db.insert_currency_data(data_1d, p,"currency")
        db.insert_currency_data(data_4h, p,"currency")
        db.insert_currency_data(data_1h, p,"currency")
        # db.insert_currency_data(data_15m, p,"currency")
        # db.insert_currency_data(data_5m, p,"currency")
        # db.insert_currency_data(currency, p,"currency")

    # 2. Salva le coppie nel DB (Operazione una tantum o di aggiornamento)

    # db.insert_all_pairs(all_pairs_eur)
    db.close_connection()
    print("\n=== FINE ===")

def main4():
    market_prov = MarketDataProvider()
    forecast = TimeSfmForecaster()
    db = DatabaseManager()
    all_pairs_eur = market_prov.getAllPairs(quote_filter="EUR", leverage_only=True)
    offset = 30
    while offset > 1:
        for p in all_pairs_eur:
            data_1d = db.get_candles_with_offset('currency', "1d", p['base'], offset)
            data_4h =db.get_candles_with_offset('currency', "4h", p['base'], offset)
            data_1h =db.get_candles_with_offset('currency', "1h", p['base'], offset)
            data_15m = db.get_candles_with_offset('currency', "15m", p['base'], offset)
            data_5m = db.get_candles_with_offset('currency', "5m", p['base'], offset)
            currency = db.get_candles_with_offset('currency', "1m", p['base'], offset)
            # res = forecast.predict_candles(data_1d, "1d", 2, p)
            # res1 = forecast.predict_candles(data_4h, "4h", 2, p)
            # res5 = forecast.predict_candles(data_1h, "1h", 2, p)
            res2 = forecast.predict_candles(data_15m, "15m", 2, p)
            res3 = forecast.predict_candles(data_5m, "5m", 2, p)
            res4 = forecast.predict_candles(currency, "1m", 2, p)
            # db.insert_currency_data(res, p,"forecast")
            # db.insert_currency_data(res1, p,"forecast")
            db.insert_currency_data(res2, p,"forecast")
            db.insert_currency_data(res3, p,"forecast")
            db.insert_currency_data(res4, p,"forecast")
            # db.insert_currency_data(res5, p,"forecast")

        offset= offset - 1
    db.close_connection()

def main5():
    market_prov = MarketDataProvider()
    forecast = TimeSfmForecaster()
    db = DatabaseManager()
    all_pairs_eur = market_prov.getAllPairs(quote_filter="EUR", leverage_only=True)
    timeD = "2025-11-29 12:00:00"
    finalD = "2025-11-30 12:00:00"
    timeframes = {
        "1d": timeD,
        "4h": timeD,
        "1h": timeD,
        "15m": timeD,
        "5m": timeD,
        "1m": timeD
    }
    offset = True
    offset1m, offset5m, offset15m, offset1h, offset4h, offset1d = True, True, True, True, True, True
    while offset == True:
        for p in all_pairs_eur:
            if offset1d == True:
                data_1d = db.get_candles_before_date('currency', "1d", p['base'], timeframes["1d"])
                res = forecast.predict_candles(data_1d, "1d", 2, p)
                db.insert_currency_data(res, p,"forecast")
            if offset4h == True:
                data_4h =db.get_candles_before_date('currency', "4h", p['base'], timeframes["4h"])
                res1 = forecast.predict_candles(data_4h, "4h", 3, p)
                db.insert_currency_data(res1, p,"forecast")
            if offset1h == True:
                data_1h =db.get_candles_before_date('currency', "1h", p['base'], timeframes["1h"])
                res5 = forecast.predict_candles(data_1h, "1h", 4, p)
                db.insert_currency_data(res5, p,"forecast")
            if offset15m == True:
                data_15m = db.get_candles_before_date('currency', "15m", p['base'], timeframes["15m"])
                res2 = forecast.predict_candles(data_15m, "15m", 6, p)
                db.insert_currency_data(res2, p,"forecast")
            # if offset5m == True:
            #     data_5m = db.get_candles_before_date('currency', "5m", p['base'], timeframes["5m"])
            #     res3 = forecast.predict_candles(data_5m, "5m", 2, p)
            #     db.insert_currency_data(res3, p,"forecast")
            # if offset1m == True:
            #     currency = db.get_candles_before_date('currency', "1m", p['base'], timeframes["1m"])
            #     res4 = forecast.predict_candles(currency, "1m", 2, p)
            #     db.insert_currency_data(res4, p,"forecast")

        timeframes["1d"] = db.add_timeframe(timeframes["1d"], "1d") if offset1d == True else timeframes["1d"]
        timeframes["4h"] = db.add_timeframe(timeframes["4h"], "4h") if offset4h == True else timeframes["4h"]
        timeframes["1h"] = db.add_timeframe(timeframes["1h"], "1h") if offset1h == True else timeframes["1h"]
        timeframes["15m"] = db.add_timeframe(timeframes["15m"], "15m") if offset15m == True else timeframes["15m"]
        # timeframes["5m"] = db.add_timeframe(timeframes["5m"], "5m") if offset5m == True else timeframes["5m"]
        # timeframes["1m"] = db.add_timeframe(timeframes["1m"], "1m") if offset1m == True else timeframes["1m"]

        # offset1m, offset5m, offset15m, offset1h, offset4h, offset1d = db.is_after(finalD, timeframes["1m"]), db.is_after(finalD, timeframes["5m"]), db.is_after(finalD, timeframes["15m"]), db.is_after(finalD, timeframes["1h"]), db.is_after(finalD, timeframes["4h"]), db.is_after(finalD, timeframes["1d"])
        offset15m, offset1h, offset4h, offset1d = db.is_after(finalD, timeframes["15m"]), db.is_after(finalD, timeframes["1h"]), db.is_after(finalD, timeframes["4h"]), db.is_after(finalD, timeframes["1d"])
        if offset15m == False and offset1h == False and offset4h == False and offset1d == False:
            offset = False

        # if offset1m == False and offset5m == False and offset15m == False and offset1h == False and offset4h == False and offset1d == False:
        #     offset = False

        print(f'offset1m: {offset1m} timeframe1m: {timeframes["1m"]}')
        print(f'offset5m: {offset5m} timeframe5m: {timeframes["5m"]}')
        print(f'offset15m: {offset15m} timeframe15m: {timeframes["15m"]}')
        print(f'offset1h: {offset1h} timeframe1h: {timeframes["1h"]}')
        print(f'offset4h: {offset4h} timeframe4h: {timeframes["4h"]}')
        print(f'offset1d: {offset1d} timeframe1d: {timeframes["1d"]}')

    db.close_connection()

if __name__ == "__main__":
    # main2()
    # main3()
    # main()
    # main4()
    main5()
