#!/usr/bin/env python3
"""
Performance Explainer - Comprehensive Analysis
Portfolio PnL calculation with Bloomberg data fetching
"""

import pandas as pd
import blpapi
from blpapi import SessionOptions, Session
import os
from datetime import datetime
import numpy as np
from openai import OpenAI
import json
import requests

class PerformanceExplainer:
    def __init__(self, gmail_script_url=None):
        """Initialize the Performance Explainer"""
        self.session = None
        self.service_uri = "//blp/refdata"
        self.openai_client = OpenAI(api_key="APIKEYPLACEHOLDER")
        self.gmail_script_url = gmail_script_url
        
        # Initialize Bloomberg session
        self.init_bloomberg_session()
    
    def init_bloomberg_session(self):
        """Initialize Bloomberg Terminal session"""
        try:
            self.session = blpapi.Session()
            if not self.session.start():
                raise Exception("Failed to start Bloomberg session.")
            if not self.session.openService(self.service_uri):
                raise Exception("Failed to open refdata service.")
            print("Bloomberg Terminal session initialized")
        except Exception as e:
            print(f"Bloomberg connection failed: {e}")
            print("Make sure Bloomberg Terminal is running and logged in.")
            self.session = None
    
    def parse_amount(self, value):
        """Parse amount string, handling parentheses as negative and commas"""
        if isinstance(value, (int, float)):
            return value
        value = str(value).strip().replace(",", "")
        if value.startswith("(") and value.endswith(")"):
            value = "-" + value[1:-1] 
        return float(value)
    
    def fetch_bbg_history(self, tickers, start_date, end_date, field="PX_LAST"):
        """Fetch historical data for multiple tickers"""
        if not self.session:
            print("No Bloomberg session available")
            return pd.DataFrame()
            
        try:
            service = self.session.getService(self.service_uri)
            request = service.createRequest("HistoricalDataRequest")
            
            for ticker in tickers:
                request.append("securities", ticker)
            request.append("fields", field)
            request.set("startDate", start_date)
            request.set("endDate", end_date)
            request.set("periodicitySelection", "DAILY")

            self.session.sendRequest(request)
            all_data = []
            
            while True:
                event = self.session.nextEvent()
                for msg in event:
                    if msg.hasElement("securityData"):
                        ticker = msg.getElement("securityData").getElementAsString("security")
                        data = msg.getElement("securityData").getElement("fieldData")
                        for i in range(data.numValues()):
                            item = data.getValue(i)
                            row = {"ticker": ticker, "date": item.getElementAsDatetime("date")}
                            value = item.getElementAsFloat(field) if item.hasElement(field) else None
                            row[field] = value
                            all_data.append(row)
                if event.eventType() == blpapi.Event.RESPONSE:
                    break
                    
            return pd.DataFrame(all_data)
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return pd.DataFrame()
    
    def create_output_folder(self):
        """Create outputs folder if it doesn't exist"""
        output_dir = "outputs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created '{output_dir}' folder")
        return output_dir
    
    def generate_smart_search_terms(self, ticker):
        """Use AI to generate smart search terms for a ticker"""
        try:
            # Create a prompt to generate relevant search terms
            prompt = f"""
            For the financial instrument "{ticker}", generate a list of relevant search terms that would help find related news headlines in financial emails.
            
            Consider:
            - Company names, subsidiaries, and business segments
            - Industry sectors and related companies
            - Geographic regions and markets
            - Economic indicators and events that might affect this instrument
            - Alternative ticker symbols and naming variations
            
            Return ONLY a comma-separated list of 8-12 search terms, no explanations.
            Example format: Apple, AAPL, iPhone, Tim Cook, technology sector, smartphone sales
            
            Ticker: {ticker}
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a financial research assistant specializing in identifying relevant search terms for market instruments."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            # Parse the response to get search terms
            search_terms_text = response.choices[0].message.content.strip()
            search_terms = [term.strip() for term in search_terms_text.split(',')]
            
            # Clean and filter terms
            search_terms = [term for term in search_terms if term and len(term) > 2]
            
            print(f"Generated {len(search_terms)} smart search terms for {ticker}")
            return search_terms[:12]  # Limit to 12 terms
            
        except Exception as e:
            print(f"Error generating smart search terms for {ticker}: {e}")
            # Fallback to basic terms
            base_symbol = ticker.split()[0] if ' ' in ticker else ticker
            return [base_symbol, ticker]

    def smart_gmail_search(self, ticker, start_date, end_date):
        """Enhanced Gmail search using AI-generated search terms"""
        if not self.gmail_script_url:
            return []
            
        news_data = []
        
        try:
            print(f"Performing smart Gmail search for {ticker}...")
            
            # Generate smart search terms using AI
            search_terms = self.generate_smart_search_terms(ticker)
            print(f"Search terms: {', '.join(search_terms[:5])}{'...' if len(search_terms) > 5 else ''}")
            
            # Calculate days back from date range
            start_dt = datetime.strptime(start_date, "%Y%m%d")
            days_back = min((datetime.now() - start_dt).days, 30)
            
            # Try multiple search approaches
            search_approaches = [
                # Approach 1: Primary ticker/symbol
                [ticker.split()[0] if ' ' in ticker else ticker],
                # Approach 2: Top 3 AI-generated terms
                search_terms[:3],
                # Approach 3: Company/sector terms (skip ticker symbols)
                [term for term in search_terms[1:] if not any(c.isupper() for c in term) and len(term) > 3][:3],
                # Approach 4: Broader search with any of the terms
                search_terms[:6]
            ]
            
            all_headlines = {}  # Use dict to avoid duplicates by subject
            
            for approach_num, terms in enumerate(search_approaches, 1):
                if not terms:
                    continue
                    
                try:
                    print(f"  Approach {approach_num}: Searching for {terms}")
                    
                    # Prepare parameters for Google Apps Script
                    params = {
                        'action': 'getNewsHeadlines',
                        'ticker': ' OR '.join(f'"{term}"' for term in terms),  # Search for any of the terms
                        'daysBack': days_back,
                        'maxResults': 15
                    }
                    
                    # Make request to Google Apps Script
                    response = requests.get(self.gmail_script_url, params=params, timeout=30)
                    response.raise_for_status()
                    
                    data = response.json()
                    
                    if data.get('status') == 'success':
                        gmail_headlines = data.get('headlines', [])
                        print(f"    Found {len(gmail_headlines)} headlines")
                        
                        # Process and deduplicate headlines
                        for item in gmail_headlines:
                            subject = item.get('subject', 'No subject')
                            if subject not in all_headlines:  # Avoid duplicates
                                try:
                                    # Use AI to determine relevance
                                    relevance_score = self.calculate_headline_relevance(subject, ticker, search_terms)
                                    
                                    if relevance_score > 0.3:  # Only include relevant headlines
                                        news_item = {
                                            'date': datetime.fromisoformat(item.get('date', datetime.now().isoformat()).replace('Z', '+00:00')),
                                            'headline': subject,
                                            'source': f"Gmail Smart Search ({item.get('sender', 'Unknown')})",
                                            'story': item.get('snippet', ''),
                                            'relevance_score': relevance_score,
                                            'search_approach': approach_num
                                        }
                                        all_headlines[subject] = news_item
                                        
                                except Exception as e:
                                    print(f"    Error processing headline: {e}")
                                    continue
                    else:
                        print(f"    Search failed: {data.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    print(f"  Approach {approach_num} failed: {e}")
                    continue
            
            # Convert to list and sort by relevance
            news_data = list(all_headlines.values())
            news_data.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            # Limit to top 10 most relevant
            news_data = news_data[:10]
            
            print(f"Smart search found {len(news_data)} relevant headlines for {ticker}")
            
        except Exception as e:
            print(f"Error in smart Gmail search for {ticker}: {e}")
        
        return news_data

    def calculate_headline_relevance(self, headline, ticker, search_terms):
        """Use AI to calculate how relevant a headline is to the ticker"""
        try:
            prompt = f"""
            Rate the relevance of this news headline to the financial instrument "{ticker}" on a scale of 0.0 to 1.0.
            
            Consider:
            - Direct mentions of the company/instrument
            - Industry/sector relevance
            - Market-moving events that could affect the instrument
            - Economic factors that might impact the instrument
            
            Search terms used: {', '.join(search_terms)}
            Headline: "{headline}"
            Ticker: {ticker}
            
            Return ONLY a number between 0.0 and 1.0 (e.g., 0.8)
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a financial analyst rating news relevance. Return only a decimal number."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.1
            )
            
            # Parse the relevance score
            score_text = response.choices[0].message.content.strip()
            score = float(score_text)
            return max(0.0, min(1.0, score))  # Ensure it's between 0 and 1
            
        except Exception as e:
            print(f"Error calculating relevance: {e}")
            # Fallback: simple keyword matching
            headline_lower = headline.lower()
            ticker_symbol = ticker.split()[0] if ' ' in ticker else ticker
            
            if ticker_symbol.lower() in headline_lower:
                return 0.9
            elif any(term.lower() in headline_lower for term in search_terms[:3]):
                return 0.6
            else:
                return 0.3

    def fetch_news_data(self, ticker, start_date, end_date):
        """Fetch news data from Gmail using smart AI-powered search"""
        if not self.session:
            return []
            
        news_data = []
        
        # Method 1: Smart Gmail search using AI-generated search terms
        try:
            if self.gmail_script_url:
                print(f"Fetching Gmail headlines for {ticker}...")
                
                # Use the new smart search function
                smart_headlines = self.smart_gmail_search(ticker, start_date, end_date)
                
                if smart_headlines:
                    print(f"Smart search retrieved {len(smart_headlines)} relevant headlines")
                    news_data.extend(smart_headlines)
                else:
                    print("Smart search found no relevant headlines")
                    
        except Exception as e:
            print(f"Error in smart Gmail search: {e}")
        
        # Method 2: Fallback to basic Gmail search if smart search failed or found nothing
        if not news_data and self.gmail_script_url:
            try:
                print(f"Trying basic Gmail search for {ticker}...")
                
                # Calculate days back from date range
                start_dt = datetime.strptime(start_date, "%Y%m%d")
                days_back = min((datetime.now() - start_dt).days, 30)
                
                # Prepare parameters for Google Apps Script (basic search)
                params = {
                    'action': 'getNewsHeadlines',
                    'ticker': ticker.split()[0] if ' ' in ticker else ticker,
                    'daysBack': days_back,
                    'maxResults': 10
                }
                
                # Make request to Google Apps Script
                response = requests.get(self.gmail_script_url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                if data.get('status') == 'success':
                    gmail_headlines = data.get('headlines', [])
                    print(f"Basic search retrieved {len(gmail_headlines)} headlines")
                    
                    # Convert Gmail headlines to standard format
                    for item in gmail_headlines:
                        try:
                            news_item = {
                                'date': datetime.fromisoformat(item.get('date', datetime.now().isoformat()).replace('Z', '+00:00')),
                                'headline': item.get('subject', 'No subject'),
                                'source': f"Gmail Basic Search ({item.get('sender', 'Unknown sender')})",
                                'story': item.get('snippet', ''),
                                'relevance_score': 0.5,  # Default relevance for basic search
                                'search_approach': 'basic'
                            }
                            news_data.append(news_item)
                        except Exception as e:
                            print(f"Error processing basic search headline: {e}")
                            continue
                else:
                    print(f"Basic Gmail search failed: {data.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"Error in basic Gmail search: {e}")
        
        # Method 3: Fallback to price analysis if no Gmail headlines found
        if not news_data:
            print(f"Using price analysis fallback for {ticker}...")
            try:
                # Get recent price history to analyze for significant moves
                price_data = self.fetch_bbg_history([ticker], start_date, end_date, "PX_LAST")
                
                if not price_data.empty:
                    ticker_data = price_data[price_data['ticker'] == ticker].sort_values('date')
                    if len(ticker_data) >= 2:
                        ticker_data['returns'] = ticker_data['PX_LAST'].pct_change()
                        
                        # Find significant price movements
                        std_dev = ticker_data['returns'].std()
                        if not pd.isna(std_dev) and std_dev > 0:
                            significant_moves = ticker_data[abs(ticker_data['returns']) > 2 * std_dev]
                            
                            for _, move in significant_moves.iterrows():
                                direction = "increased" if move['returns'] > 0 else "decreased"
                                magnitude = abs(move['returns'] * 100)
                                
                                news_item = {
                                    "date": move['date'],
                                    "headline": f"{ticker} {direction} {magnitude:.1f}% - significant price movement detected",
                                    "source": "Price Analysis",
                                    "relevance_score": 0.8,
                                    "search_approach": "price_analysis"
                                }
                                news_data.append(news_item)
                
            except Exception as e:
                print(f"Price analysis failed for {ticker}: {e}")
        
        # Always provide at least one context item
        if not news_data:
            news_data.append({
                "date": datetime.now(),
                "headline": f"Analysis period {start_date} to {end_date} for {ticker} - monitoring price volatility and market conditions",
                "source": "Market Analysis",
                "relevance_score": 0.3,
                "search_approach": "fallback"
            })
        
        # Sort by relevance score and limit results
        news_data.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        news_data = news_data[:15]  # Keep top 15 most relevant items
        
        print(f"Total news items for {ticker}: {len(news_data)} (avg relevance: {sum(item.get('relevance_score', 0) for item in news_data)/len(news_data):.2f})")
        return news_data
    
    def calculate_price_volatility(self, df_hist, ticker):
        """Calculate price volatility and identify extreme fluctuations"""
        ticker_data = df_hist[df_hist['ticker'] == ticker].copy()
        if ticker_data.empty or len(ticker_data) < 2:
            return None
            
        ticker_data = ticker_data.sort_values('date')
        ticker_data['returns'] = ticker_data['PX_LAST'].pct_change()
        ticker_data['abs_returns'] = ticker_data['returns'].abs()
        
        # Calculate standard deviation
        std_dev = ticker_data['returns'].std()
        mean_return = ticker_data['returns'].mean()
        
        # Define extreme fluctuations as movements > 2 standard deviations
        threshold = 2 * std_dev
        extreme_moves = ticker_data[ticker_data['abs_returns'] > threshold].copy()
        
        volatility_stats = {
            'ticker': ticker,
            'std_dev': std_dev,
            'mean_return': mean_return,
            'max_move': ticker_data['abs_returns'].max(),
            'extreme_moves': extreme_moves[['date', 'PX_LAST', 'returns', 'abs_returns']].to_dict('records') if not extreme_moves.empty else []
        }
        
        return volatility_stats
    
    def analyze_with_ai(self, ticker, volatility_stats, news_data, pnl_data):
        """Use AI to analyze price movements and correlate with news"""
        try:
            # Prepare context for AI
            context = f"""
            Ticker: {ticker}
            
            Volatility Statistics:
            - Standard Deviation: {volatility_stats['std_dev']:.4f}
            - Mean Return: {volatility_stats['mean_return']:.4f}
            - Maximum Move: {volatility_stats['max_move']:.4f}
            
            Extreme Price Movements (>2 std dev):
            {json.dumps(volatility_stats['extreme_moves'], indent=2, default=str)}
            
            PnL Impact: {pnl_data}
            
            Recent News Headlines:
            {json.dumps([{"date": str(item["date"]), "headline": item.get("headline", item.get("story", "No headline"))} for item in news_data[:10]], indent=2)}
            """
            
            prompt = f"""
            You are a financial analyst. Analyze the price movements and news for {ticker}.
            
            Context:
            {context}
            
            Please provide:
            1. Identification of the most significant price movements
            2. Correlation between news events and price volatility
            3. Potential reasons for extreme fluctuations
            4. Impact assessment on portfolio PnL
            5. Key risk factors identified
            
            Keep the analysis concise but insightful, focusing on actionable insights.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert financial analyst specializing in market volatility and news impact analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error in AI analysis for {ticker}: {e}")
            return f"AI analysis failed for {ticker}: {str(e)}"

    def generate_filename(self, start_date, end_date, file_type="analysis"):
        """Generate unique filename with timestamp"""
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        if file_type == "analysis":
            filename = f"portfolio_analysis_{start_date}_to_{end_date}_{current_time}.csv"
        elif file_type == "volatility":
            filename = f"volatility_analysis_{start_date}_to_{end_date}_{current_time}.csv"
        elif file_type == "ai_report":
            filename = f"ai_market_report_{start_date}_to_{end_date}_{current_time}.txt"
        return filename

    def get_date_input(self, prompt, default_date):
        """Get date input from user with validation"""
        while True:
            date_input = input(f"{prompt} (YYYYMMDD format, or press Enter for {default_date}): ").strip()
            
            if not date_input:
                return default_date
            
            # Validate date format
            if len(date_input) == 8 and date_input.isdigit():
                try:
                    # Test if it's a valid date
                    from datetime import datetime
                    datetime.strptime(date_input, "%Y%m%d")
                    return date_input
                except ValueError:
                    print("Invalid date. Please use YYYYMMDD format (e.g., 20250630)")
            else:
                print("Invalid format. Please use YYYYMMDD format (e.g., 20250630)")

    def test_news_access(self, ticker="AAPL US Equity"):
        """Test news data access for a ticker"""
        print(f"Testing news access for {ticker}...")
        news_data = self.fetch_news_data(ticker, "20250720", "20250724")
        
        if news_data:
            print(f"Successfully retrieved {len(news_data)} news items")
            for i, item in enumerate(news_data[:3]):
                print(f"{i+1}. [{item.get('source', 'Unknown')}] {item.get('headline', 'No headline')}")
        else:
            print("No news data retrieved")
        
        return len(news_data) > 0

    def comprehensive_analysis(self, start_date=None, end_date=None):
        """Run comprehensive analysis with historical data"""
        print("\nRunning Comprehensive Analysis...")
        
        # Get dates from user if not provided
        if start_date is None:
            print("\nEnter analysis period:")
            start_date = self.get_date_input("Start date", "20250530")
            end_date = self.get_date_input("End date", "20250630")
        
        print(f"Analysis period: {start_date} to {end_date}")
        
        # Define positions
        dv01_tickers = ["USGG5YR Index", "USGGT20Y Index", "USSMSO25 CMPN Curncy"]
        dv01_values = ["125,000", "145,000", "750,000"]
        dv01_mapping = {t: self.parse_amount(v) for t, v in zip(dv01_tickers, dv01_values)}
        
        notional_tickers = [
            "CCJ Equity", "GDX Equity", "SIA Comdty", "PLA comdty", "HGA comdty",
            "CLZ5 Comdty", "USDCNH curncy", "USDHKD curncy",
            "CHFJPY curncy", "IBIT Equity", "ESA index"
        ]
        notional_values = [
            "5,390,000", "6,100,000", "13,500,000", "16,320,000", "17,000,000", "9,600,000",
            "371,000,000", "365,000,000", "(157,000,000)", "18,830,000", "(30,000,000)"
        ]
        notional_mapping = {t: self.parse_amount(v) for t, v in zip(notional_tickers, notional_values)}

        # Fetch historical data
        all_tickers = dv01_tickers + notional_tickers
        
        df_hist = self.fetch_bbg_history(all_tickers, start_date, end_date, field="PX_LAST")
        
        if df_hist.empty:
            print("No historical data retrieved")
            return None
        
        # Calculate DV01 PnL
        dv01_pnl_data = []
        for ticker, dv01 in dv01_mapping.items():
            px = df_hist[df_hist['ticker'] == ticker].sort_values('date').set_index('date')
            if len(px) < 2 or px['PX_LAST'].isnull().all():
                continue
            px_diff = (px['PX_LAST'].diff() / 0.01)
            pnl = px_diff * dv01 * -1
            tmp = pnl.reset_index()
            tmp['ticker'] = ticker
            dv01_pnl_data.append(tmp)
        
        # Calculate Notional PnL
        notional_pnl_data = []
        for ticker, notional in notional_mapping.items():
            px = df_hist[df_hist['ticker'] == ticker].sort_values('date').set_index('date')
            if len(px) < 2 or px['PX_LAST'].isnull().all():
                continue
            px_diff = px['PX_LAST'].pct_change().dropna()
            pnl = px_diff * notional
            tmp = pnl.reset_index()
            tmp['ticker'] = ticker
            notional_pnl_data.append(tmp)
        
        # Create summary
        summary_data = []
        
        # Add DV01 positions
        for ticker in dv01_mapping.keys():
            px_data = df_hist[df_hist['ticker'] == ticker].sort_values('date')
            if px_data.empty:
                continue
            px_start, px_end = px_data['PX_LAST'].iloc[0], px_data['PX_LAST'].iloc[-1]
            if dv01_pnl_data:
                pnl_df = pd.concat(dv01_pnl_data, ignore_index=True)
                pnl = pnl_df[pnl_df['ticker'] == ticker]['PX_LAST'].sum() if not pnl_df.empty else 0
            else:
                pnl = 0
            summary_data.append([ticker, "DV01", dv01_mapping[ticker], px_start, px_end, pnl])

        # Add Notional positions
        for ticker in notional_mapping.keys():
            px_data = df_hist[df_hist['ticker'] == ticker].sort_values('date')
            if px_data.empty:
                continue
            px_start, px_end = px_data['PX_LAST'].iloc[0], px_data['PX_LAST'].iloc[-1]
            if notional_pnl_data:
                pnl_df = pd.concat(notional_pnl_data, ignore_index=True)
                pnl = pnl_df[pnl_df['ticker'] == ticker]['PX_LAST'].sum() if not pnl_df.empty else 0
            else:
                pnl = 0
            summary_data.append([ticker, "Notional", notional_mapping[ticker], px_start, px_end, pnl])

        summary_df = pd.DataFrame(summary_data, columns=["ticker", "position_type", "amount", "px_start", "px_end", "pnl"])
        
        print("\nComprehensive Analysis Results:")
        print(summary_df.to_string(index=False))
        total_pnl = summary_df['pnl'].sum()
        print(f"\nTotal Portfolio P&L: {total_pnl:,.2f}")
        
        # Create outputs folder and generate unique filenames
        output_dir = self.create_output_folder()
        pnl_filename = self.generate_filename(start_date, end_date, "analysis")
        volatility_filename = self.generate_filename(start_date, end_date, "volatility")
        ai_report_filename = self.generate_filename(start_date, end_date, "ai_report")
        
        pnl_filepath = os.path.join(output_dir, pnl_filename)
        volatility_filepath = os.path.join(output_dir, volatility_filename)
        ai_report_filepath = os.path.join(output_dir, ai_report_filename)
        
        # Save PnL analysis to CSV
        summary_df.to_csv(pnl_filepath, index=False)
        print(f"PnL Results saved to {pnl_filepath}")
        
        # Perform volatility analysis and AI correlation
        print("\nAnalyzing Price Volatility and News Correlation...")
        volatility_data = []
        ai_analyses = []
        
        for ticker in all_tickers:
            print(f"Analyzing {ticker}...")
            
            # Calculate volatility statistics
            vol_stats = self.calculate_price_volatility(df_hist, ticker)
            if vol_stats is None:
                continue
                
            # Get PnL for this ticker
            ticker_pnl = summary_df[summary_df['ticker'] == ticker]['pnl'].iloc[0] if not summary_df[summary_df['ticker'] == ticker].empty else 0
            
            # Fetch news data
            news_data = self.fetch_news_data(ticker, start_date, end_date)
            
            # AI analysis
            ai_analysis = self.analyze_with_ai(ticker, vol_stats, news_data, ticker_pnl)
            
            # Prepare volatility data for CSV
            vol_record = {
                'ticker': ticker,
                'std_dev': vol_stats['std_dev'],
                'mean_return': vol_stats['mean_return'],
                'max_move': vol_stats['max_move'],
                'num_extreme_moves': len(vol_stats['extreme_moves']),
                'pnl_impact': ticker_pnl,
                'risk_level': 'HIGH' if vol_stats['std_dev'] > 0.02 else 'MEDIUM' if vol_stats['std_dev'] > 0.01 else 'LOW'
            }
            volatility_data.append(vol_record)
            
            # Store AI analysis
            ai_analyses.append({
                'ticker': ticker,
                'analysis': ai_analysis,
                'extreme_moves': vol_stats['extreme_moves']
            })
        
        # Save volatility analysis to CSV
        volatility_df = pd.DataFrame(volatility_data)
        volatility_df.to_csv(volatility_filepath, index=False)
        print(f"Volatility Analysis saved to {volatility_filepath}")
        
        # Generate comprehensive AI report
        print("\nGenerating AI Market Analysis Report...")
        
        with open(ai_report_filepath, 'w', encoding='utf-8') as f:
            f.write("AI-POWERED MARKET ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Period: {start_date} to {end_date}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Portfolio PnL: {total_pnl:,.2f}\n\n")
            
            # Executive Summary
            high_risk_tickers = [item for item in volatility_data if item['risk_level'] == 'HIGH']
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total positions analyzed: {len(all_tickers)}\n")
            f.write(f"High volatility positions: {len(high_risk_tickers)}\n")
            f.write(f"Average portfolio volatility: {np.mean([item['std_dev'] for item in volatility_data]):.4f}\n\n")
            
            # Individual ticker analyses
            f.write("DETAILED TICKER ANALYSIS\n")
            f.write("-" * 25 + "\n\n")
            
            for analysis in ai_analyses:
                f.write(f"{analysis['ticker']}\n")
                f.write("-" * len(analysis['ticker']) + "---\n")
                f.write(f"{analysis['analysis']}\n\n")
                
                if analysis['extreme_moves']:
                    f.write("Extreme Price Movements:\n")
                    for move in analysis['extreme_moves']:
                        f.write(f"  {move['date']}: {move['returns']:.4f} ({move['returns']*100:.2f}%)\n")
                    f.write("\n")
                
                f.write("-" * 50 + "\n\n")
            
            # Risk Summary
            f.write("RISK ASSESSMENT SUMMARY\n")
            f.write("-" * 25 + "\n")
            for item in sorted(volatility_data, key=lambda x: x['std_dev'], reverse=True):
                f.write(f"{item['ticker']}: {item['risk_level']} risk (σ={item['std_dev']:.4f}, PnL={item['pnl_impact']:,.0f})\n")
        
        print(f"AI Market Report saved to {ai_report_filepath}")
        
        # Also save a summary file with just the totals
        summary_info = {
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "period_start": start_date,
            "period_end": end_date,
            "total_pnl": total_pnl,
            "num_positions": len(summary_df),
            "high_volatility_positions": len(high_risk_tickers),
            "pnl_file": pnl_filename,
            "volatility_file": volatility_filename,
            "ai_report_file": ai_report_filename
        }
        
        summary_filename = f"summary_{start_date}_to_{end_date}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        summary_filepath = os.path.join(output_dir, summary_filename)
        
        with open(summary_filepath, 'w') as f:
            f.write("Portfolio Analysis Summary\n")
            f.write("=" * 30 + "\n")
            f.write(f"Analysis Date: {summary_info['analysis_date']}\n")
            f.write(f"Period: {summary_info['period_start']} to {summary_info['period_end']}\n")
            f.write(f"Total PnL: {summary_info['total_pnl']:,.2f}\n")
            f.write(f"Number of Positions: {summary_info['num_positions']}\n")
            f.write(f"High Volatility Positions: {summary_info['high_volatility_positions']}\n\n")
            f.write("Generated Files:\n")
            f.write(f"PnL Analysis: {summary_info['pnl_file']}\n")
            f.write(f"Volatility Analysis: {summary_info['volatility_file']}\n")
            f.write(f"AI Market Report: {summary_info['ai_report_file']}\n")
        
        print(f"Summary saved to {summary_filepath}")
        print(f"\nAll files saved in '{output_dir}' folder")
        
        return summary_df
    
    def __del__(self):
        """Cleanup Bloomberg session"""
        if self.session:
            self.session.stop()

def main():
    """Main function"""
    print("Performance Explainer - Comprehensive Analysis")
    print("="*60)
    
    # Set your Google Apps Script URL here (replace with your deployed script URL)
    gmail_script_url = "https://script.google.com/macros/s/AKfycbzUYxsavp80l2DC7G3S51q_B3yvGwBy7mIVagATZuhYNq6KnY_AMkOI1nHTIVbf_2gV/exec"
    
    # Create and run analyzer
    analyzer = PerformanceExplainer(gmail_script_url=gmail_script_url)
    
    if analyzer.session:
        if gmail_script_url:
            print("Gmail integration enabled")
        else:
            print("Gmail integration disabled - using price analysis only")
        
        analyzer.comprehensive_analysis()
    else:
        print("Cannot run analysis without Bloomberg connection")

if __name__ == "__main__":
    main()