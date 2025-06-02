# data_loader.py
import yfinance as yf
import pandas as pd
import time
import requests
from datetime import datetime

def get_sp500_tickers():
    """Récupère la liste des tickers S&P500 depuis Wikipedia"""
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        table = pd.read_html(url)[0]
        tickers = table['Symbol'].tolist()
        # Nettoyer les tickers pour yfinance
        tickers = [t.replace('.', '-') for t in tickers]
        print(f"Récupéré {len(tickers)} tickers S&P500")
        return tickers
    except Exception as e:
        print(f"Erreur lors de la récupération des tickers S&P500: {e}")
        return []

def get_additional_tickers():
    """Ajoute des tickers populaires pour atteindre ~500 entreprises"""
    additional_tickers = [
        # Tech non-S&P500
        'TSLA', 'NVDA', 'PLTR', 'SNOW', 'COIN', 'RBLX', 'HOOD', 'SOFI',
        # International
        'BABA', 'TSM', 'ASML', 'NVO', 'UL', 'TM', 'SONY', 'SAP', 'SHOP',
        # Crypto-related
        'MSTR', 'RIOT', 'MARA', 'CLSK',
        # ETFs populaires
        'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'GLD', 'SLV',
        # REIT
        'O', 'PLD', 'AMT', 'CCI', 'EQIX', 'PSA', 'WELL', 'AVB', 'EXR',
        # Autres secteurs
        'BRK-B', 'BRK-A', 'UBER', 'LYFT', 'DASH', 'ABNB', 'SPOT', 'NFLX',
        'DIS', 'PYPL', 'SQ', 'ROKU', 'ZM', 'DOCU', 'CRWD', 'ZS', 'OKTA'
    ]
    return additional_tickers

def validate_ticker(ticker):
    """Vérifie qu'un ticker existe et a des données"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        return len(hist) > 0
    except:
        return False

def download_price_data_batch(tickers, start="2018-01-01", end="2023-01-01", interval='1mo', max_retries=3):
    """Télécharge les prix par petits lots pour éviter les timeouts"""
    all_data = pd.DataFrame()
    batch_size = 20  # Réduire la taille des lots pour plus de stabilité
    
    print(f"Téléchargement de {len(tickers)} tickers entre {start} et {end}...")
    
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        batch_num = i//batch_size + 1
        total_batches = (len(tickers) + batch_size - 1) // batch_size
        
        print(f"Traitement du lot {batch_num}/{total_batches} ({len(batch)} tickers)...")
        print(f"Tickers: {batch[:5]}{'...' if len(batch) > 5 else ''}")  # Afficher quelques tickers pour debug
        
        retries = 0
        while retries < max_retries:
            try:
                # Télécharger le lot
                data = yf.download(
                    batch, 
                    start=start, 
                    end=end, 
                    interval=interval,
                    auto_adjust=True,
                    prepost=False,
                    threads=False,  # Désactiver le multithreading pour plus de stabilité
                    progress=False  # Désactiver la barre de progression
                )
                
                if data.empty:
                    print(f"Aucune donnée pour le lot {batch_num}")
                    break
                
                # Extraire les prix de clôture ajustés
                if len(batch) == 1:
                    # Un seul ticker
                    ticker = batch[0]
                    if not data.empty and 'Adj Close' in data.columns:
                        prices = data[['Adj Close']].rename(columns={'Adj Close': ticker})
                    elif not data.empty and 'Close' in data.columns:
                        prices = data[['Close']].rename(columns={'Close': ticker})
                    else:
                        prices = pd.DataFrame()
                else:
                    # Plusieurs tickers - yfinance retourne un MultiIndex
                    prices = pd.DataFrame()
                    if not data.empty and isinstance(data.columns, pd.MultiIndex):
                        # Structure: ('Adj Close', 'TICKER') ou ('Close', 'TICKER')
                        for ticker in batch:
                            try:
                                if ('Adj Close', ticker) in data.columns:
                                    prices[ticker] = data[('Adj Close', ticker)]
                                elif ('Close', ticker) in data.columns:
                                    prices[ticker] = data[('Close', ticker)]
                            except KeyError:
                                continue
                    elif not data.empty:
                        # Fallback si pas de MultiIndex
                        if 'Adj Close' in data.columns:
                            prices = data[['Adj Close']]
                        elif 'Close' in data.columns:
                            prices = data[['Close']]
                
                # Combiner avec les données précédentes
                if all_data.empty:
                    all_data = prices
                else:
                    all_data = pd.concat([all_data, prices], axis=1)
                
                print(f"✓ Lot {batch_num} téléchargé ({prices.shape[1]} tickers valides)")
                break
                
            except Exception as e:
                retries += 1
                print(f"Erreur lot {batch_num}, tentative {retries}/{max_retries}: {e}")
                if retries < max_retries:
                    time.sleep(2)
                else:
                    print(f"✗ Échec du lot {batch_num} après {max_retries} tentatives")
        
        # Pause entre les lots pour éviter les limites de taux
        if i + batch_size < len(tickers):
            time.sleep(2)  # Pause plus longue
    
    return all_data

def clean_data(data):
    """Nettoie les données téléchargées"""
    print("Nettoyage des données...")
    
    # Supprimer les colonnes entièrement vides
    data = data.dropna(axis=1, how='all')
    
    # Supprimer les lignes entièrement vides
    data = data.dropna(axis=0, how='all')
    
    # Trier par date
    data = data.sort_index()
    
    print(f"Données finales: {data.shape[0]} dates x {data.shape[1]} tickers")
    return data

def compute_returns(price_data):
    """Calcule les rendements mensuels"""
    print("Calcul des rendements...")
    
    # Calculer les rendements
    returns = price_data.pct_change()
    
    # Afficher des infos de debug
    print(f"Forme des prix: {price_data.shape}")
    print(f"Forme des rendements avant nettoyage: {returns.shape}")
    print(f"Nombre de valeurs non-nulles par colonne (premiers 5):")
    non_null_counts = returns.count()
    print(non_null_counts.head())
    
    # Supprimer seulement les lignes où TOUS les rendements sont NaN
    returns_cleaned = returns.dropna(how='all')
    
    # Afficher le résultat final
    print(f"Forme des rendements après nettoyage: {returns_cleaned.shape}")
    print(f"Première date avec des rendements: {returns_cleaned.index[0] if len(returns_cleaned) > 0 else 'Aucune'}")
    print(f"Dernière date avec des rendements: {returns_cleaned.index[-1] if len(returns_cleaned) > 0 else 'Aucune'}")
    
    return returns_cleaned

def save_to_csv(data, filename):
    """Sauvegarde les données en CSV"""
    if data.empty:
        print(f"⚠️  Attention: DataFrame vide pour {filename}")
        return
    
    data.to_csv(filename, index=True)
    print(f"✓ Données sauvegardées dans {filename}")
    print(f"Dimensions: {data.shape[0]} lignes x {data.shape[1]} colonnes")
    
    # Afficher un aperçu des données
    print("Aperçu des données:")
    print(data.head(3))

def main():
    """Fonction principale"""
    print("=== Récupération des données financières ===")
    
    # 1. Récupérer les tickers
    sp500_tickers = get_sp500_tickers()
    additional_tickers = get_additional_tickers()
    
    # Combiner et dédupliquer
    all_tickers = list(set(sp500_tickers + additional_tickers))
    print(f"Total de {len(all_tickers)} tickers uniques")
    
    # Limiter à 500 si nécessaire
    if len(all_tickers) > 500:
        all_tickers = all_tickers[:500]
        print(f"Limité à {len(all_tickers)} tickers")
    
    # 2. Télécharger les données
    price_data = download_price_data_batch(
        all_tickers,
        start="2008-01-01",
        end="2018-01-01",
        interval='1mo'  # 1mo pour mensuel dans yfinance
    )
    
    if price_data.empty:
        print("Aucune donnée récupérée!")
        return
    
    # 3. Nettoyer les données
    clean_prices = clean_data(price_data)
    
    # 4. Calculer les rendements
    returns = compute_returns(clean_prices)
    
    # 5. Sauvegarder
    save_to_csv(clean_prices, "monthly_prices.csv")
    
    # Ne sauvegarder les rendements que s'ils existent
    if not returns.empty:
        save_to_csv(returns, "monthly_returns.csv")
    else:
        print("⚠️  Aucun rendement calculé - fichier monthly_returns.csv non créé")
    
    print("\n=== Résumé ===")
    print(f"Période: 2018-01-01 à 2023-01-01")
    print(f"Fréquence: Mensuelle")
    print(f"Tickers récupérés: {clean_prices.shape[1]}")
    print(f"Points de données: {clean_prices.shape[0]}")
    print(f"Fichiers créés: monthly_prices.csv, monthly_returns.csv")

if __name__ == "__main__":
    main()