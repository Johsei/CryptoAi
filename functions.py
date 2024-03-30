import pandas as pd
import os
from tqdm import tqdm
import warnings
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def import_orderbooks(input_path):
    warnings.simplefilter("ignore", category=FutureWarning)

    df = pd.DataFrame()  # Leeres DataFrame erstellen
    
    print("-> reading orderbook files...")

    # Fortschrittsanzeige initialisieren
    fileCount = sum([len(files) for r, d, files in os.walk(input_path)])
    progress_bar = tqdm(total=fileCount, unit='file')

    # Rekursiv durch alle Unterordner iterieren
    for root, dirs, files in os.walk(input_path):
        # Unterordner & Dateien alphabetisch sortieren
        dirs.sort()
        files.sort()

        for file in files:
            file_path = os.path.join(root, file)
            
            # Datei in ein temporäres DataFrame einlesen
            try:
                # Einlesen der ndjson-Datei
                temp_df = pd.read_json(file_path, lines=True)

                temp_df['E'] = temp_df['E'].astype(float)

                temp_df['timestamp'] = pd.to_datetime(temp_df['E'], unit='ms')

                # Setze den Zeitstempel als Index 
                temp_df.set_index('timestamp', inplace=True)

                # Definiere das Intervall für die Kerzen, z.B. '1Min' für 1-Minuten-Kerzen
                interval = '1s'

                # Gruppiere die Daten nach dem Intervall und aggregiere zu OHLCV
                ohlcv = temp_df.groupby(pd.Grouper(freq=interval)).agg(
                    {
                        'p': ['first', 'max', 'min', 'last'], 
                        'q': 'sum'
                    }
                )

                # Benenne die Spalten um
                ohlcv.columns = ['open', 'high', 'low', 'close', 'volume']

                # Zeige das Ergebnis
                #print(ohlcv.head())

            except Exception as e:
                if not "utf-8' codec can't decode byte 0x80 in position 3131" in str(e):
                    print(f"Fehler beim Lesen der Datei: {file_path}: {e}")
                progress_bar.update(1)
                continue
            
            # Temporäres DataFrame an das Hauptdataframe anfügen
            df = pd.concat([df, ohlcv], axis=0)
            progress_bar.update(1)
            #print(df)

    progress_bar.close()
    return df

def backtesting(model, data, forecast_horizon, scaler):
    # Initialisiere die Variablen
    balance = 1000
    position = 0
    trades = 0
    win_trades = 0
    loss_trades = 0
    last_signal = 0
    signals_all = []

    # Iteriere über alle Zeilen im DataFrame
    for i in range(len(data)):
        row = data[i]
        # Berechne die Signale des Modells
        signal = model.predict(row.reshape(1, 200, 5))
        signals_all.append(signal)
        print("SIGNAL:", signal)

        # Wenn das Signal von LONG auf SHORT oder umgekehrt wechselt
        if signal != last_signal:
            # Wenn das Signal LONG ist
            if signal > 0.0001:
                # Wenn wir bereits eine SHORT-Position haben
                if position == -1:
                    # Verkaufe die SHORT-Position
                    balance += scaler.inverse_transform(row)[3] * 1
                    trades += 1
                    print("TRADED")
                    position = 0
                # Wenn wir keine LONG-Position haben
                if position == 0:
                    # Kaufe eine LONG-Position
                    position = 1
            # Wenn das Signal SHORT ist
            if signal < -0.0001:
                # Wenn wir bereits eine LONG-Position haben
                if position == 1:
                    # Verkaufe die LONG-Position
                    balance += scaler.inverse_transform(row)[3] * -1
                    trades += 1
                    print("TRADED")
                    position = 0
                # Wenn wir keine SHORT-Position haben
                if position == 0:
                    # Kaufe eine SHORT-Position
                    position = -1

        # Speichere das letzte Signal
        last_signal = signal

        # Überprüfe, ob der Trade gewonnen oder verloren wurde
        # if position == 0:
        #     if signal > 0.5 and row[3] > row[0]:
        #         win_trades += 1
        #     elif signal < 0.5 and row[3] < row[0]:
        #         win_trades += 1
        #     else:
        #         loss_trades += 1

    # Berechne den Gewinn oder Verlust
    pnl = balance - 1000

    # Berechne die Anzahl der gewonnenen und verlorenen Trades
    # win_rate = win_trades / trades * 100
    # loss_rate = loss_trades / trades * 100

    return pnl, signals_all