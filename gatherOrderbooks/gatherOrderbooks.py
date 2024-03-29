import json
import os
import websocket
from time import time
from datetime import datetime, timedelta

def check_if_already_running():
    # Aktuelle Zeit holen
    jetzt = datetime.now()

    # Zeitdifferenz definieren (10 Minuten)
    zeitdifferenz = timedelta(minutes=2)

    # Neueste Datei und deren Zeitstempel initialisieren
    neueste_datei = None
    neueste_datei_zeit = datetime.min
    
    date = datetime.now().strftime('%d-%b-%Y')
    filedir = f'/volume1/misc/CryptoAI/GatherBinanceData/RawInputData/{date}/'
    if not os.path.exists(filedir):
        return False
    
    # Durch das Verzeichnis iterieren und die neueste Datei finden
    for dateiname in os.listdir(filedir):
        dateipfad = os.path.join(filedir, dateiname)
        # Zeitstempel der letzten Änderung der Datei holen
        datei_zeit = datetime.fromtimestamp(os.path.getmtime(dateipfad))
        # Überprüfen, ob diese Datei neuer ist als die bisher neueste gefundene Datei
        if datei_zeit > neueste_datei_zeit:
            neueste_datei = dateiname
            neueste_datei_zeit = datei_zeit

    # Überprüfen, ob die neueste Datei älter als 10 Minuten ist
    if neueste_datei and jetzt - neueste_datei_zeit > zeitdifferenz:
        print(f"Die neueste Datei '{neueste_datei}' ist älter als 2 Minuten.")
        return False
    elif not neueste_datei:
        print(f"Es gibt noch keine Datei im Ordner('{neueste_datei}').")
        return False
    else:
        print(f"Die neueste Datei '{neueste_datei}' ist nicht älter als 2 Minuten.")
        return True
    

def on_message(self, message):
    d = json.loads(message)
    ws.header.append(0)
    dt = str(time())
    d['datetime'] = dt
    date = datetime.now().strftime('%d-%b-%Y')
    filedir = f'/volume1/misc/CryptoAI/GatherBinanceData/RawInputData/{date}/'
    if not os.path.exists(filedir):
        os.mkdir(filedir)
    filepath = os.path.join(filedir,f'orderbook_ws_{len(ws.header)//100_000}.ndjson')
    with open(file=filepath, mode='a+', encoding='utf-8') as f:
        json.dump(d, f)
        f.write('\n')


def on_open(self):
    print("opened")
    date = datetime.now().strftime('%d-%b-%Y')
    filedir = f'/volume1/misc/CryptoAI/GatherBinanceData/RawInputData/{date}/'
    if not os.path.exists(filedir):
        os.mkdir(filedir)
    subscribe_message = {
        "method": "SUBSCRIBE",
        "params":
        [
          "btcusdt@aggTrade",
         "btcusdt@depth10@100ms"
         ],
        "id": 1
        }

    ws.send(json.dumps(subscribe_message))

            
def on_close(self):
    print("closed connection")


if not check_if_already_running():
    print("Websocket is not running, starting...")
    socket='wss://stream.binance.com:9443/ws'

    ws = websocket.WebSocketApp(socket,
                                on_open=on_open,
                                on_message=on_message,
                                on_close=on_close)

    ws.run_forever()
else:
    print("Websocket is already running, ending script.")