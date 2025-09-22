#!/usr/bin/env python3
"""
Real-Time Data Processing Pipeline
==================================

This module provides real-time data processing capabilities including:
- Streaming data ingestion
- News sentiment analysis
- Economic calendar integration
- Social media sentiment analysis
- Market microstructure analysis
- Real-time feature engineering
"""

import pandas as pd
import numpy as np
import logging
import asyncio
import aiohttp
import json
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import queue
import time
import warnings
warnings.filterwarnings('ignore')

# News and sentiment analysis
try:
    from textblob import TextBlob
    import requests
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False

# WebSocket support
try:
    import websockets
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class MarketEvent:
    """Market event data structure"""
    timestamp: datetime
    event_type: str  # 'price', 'news', 'economic', 'social'
    symbol: str
    data: Dict[str, Any]
    source: str
    confidence: float = 1.0

@dataclass
class NewsEvent:
    """News event data structure"""
    timestamp: datetime
    title: str
    content: str
    source: str
    symbol: Optional[str] = None
    sentiment_score: float = 0.0
    impact_level: str = 'low'  # 'low', 'medium', 'high'
    category: str = 'general'

@dataclass
class EconomicEvent:
    """Economic calendar event"""
    timestamp: datetime
    country: str
    event: str
    actual: Optional[float] = None
    forecast: Optional[float] = None
    previous: Optional[float] = None
    impact: str = 'low'  # 'low', 'medium', 'high'
    currency: str = 'USD'

class RealTimeDataPipeline:
    """Real-Time Data Processing Pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the real-time data pipeline"""
        self.config = config
        
        # Data sources configuration
        self.price_sources = config.get('price_sources', {})
        self.news_sources = config.get('news_sources', {})
        self.economic_calendar = config.get('economic_calendar', {})
        self.social_media = config.get('social_media', {})
        
        # Processing configuration
        self.update_frequency = config.get('update_frequency', 1.0)  # seconds
        self.buffer_size = config.get('buffer_size', 1000)
        self.max_latency = config.get('max_latency', 0.1)  # seconds
        
        # Data buffers
        self.price_buffer = queue.Queue(maxsize=self.buffer_size)
        self.news_buffer = queue.Queue(maxsize=self.buffer_size)
        self.economic_buffer = queue.Queue(maxsize=self.buffer_size)
        self.social_buffer = queue.Queue(maxsize=self.buffer_size)
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {
            'price': [],
            'news': [],
            'economic': [],
            'social': []
        }
        
        # State tracking
        self.is_running = False
        self.last_update = {}
        self.data_cache = {}
        
        # Performance metrics
        self.metrics = {
            'events_processed': 0,
            'average_latency': 0.0,
            'errors': 0,
            'last_error': None
        }
        
        logger.info("Real-Time Data Pipeline initialized")
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """Add event handler for specific event type"""
        if event_type in self.event_handlers:
            self.event_handlers[event_type].append(handler)
            logger.info(f"Added handler for {event_type} events")
    
    def start(self):
        """Start the real-time data pipeline"""
        if self.is_running:
            logger.warning("Pipeline is already running")
            return
        
        self.is_running = True
        
        # Start processing threads
        self.price_thread = threading.Thread(target=self._process_price_data, daemon=True)
        self.news_thread = threading.Thread(target=self._process_news_data, daemon=True)
        self.economic_thread = threading.Thread(target=self._process_economic_data, daemon=True)
        self.social_thread = threading.Thread(target=self._process_social_data, daemon=True)
        
        # Start event processing thread
        self.event_thread = threading.Thread(target=self._process_events, daemon=True)
        
        # Start threads
        self.price_thread.start()
        self.news_thread.start()
        self.economic_thread.start()
        self.social_thread.start()
        self.event_thread.start()
        
        logger.info("Real-Time Data Pipeline started")
    
    def stop(self):
        """Stop the real-time data pipeline"""
        self.is_running = False
        logger.info("Real-Time Data Pipeline stopped")
    
    def _process_price_data(self):
        """Process real-time price data"""
        while self.is_running:
            try:
                # Simulate price data (in practice, connect to real data feeds)
                price_data = self._get_latest_prices()
                
                for symbol, data in price_data.items():
                    event = MarketEvent(
                        timestamp=datetime.now(),
                        event_type='price',
                        symbol=symbol,
                        data=data,
                        source='price_feed'
                    )
                    
                    self.price_buffer.put(event, timeout=1)
                
                time.sleep(self.update_frequency)
                
            except Exception as e:
                logger.error(f"Error processing price data: {e}")
                self.metrics['errors'] += 1
                self.metrics['last_error'] = str(e)
                time.sleep(1)
    
    def _process_news_data(self):
        """Process news data and sentiment analysis"""
        while self.is_running:
            try:
                # Get latest news
                news_data = self._fetch_news()
                
                for news in news_data:
                    # Analyze sentiment
                    sentiment_score = self._analyze_sentiment(news['content'])
                    
                    news_event = NewsEvent(
                        timestamp=datetime.now(),
                        title=news['title'],
                        content=news['content'],
                        source=news['source'],
                        symbol=news.get('symbol'),
                        sentiment_score=sentiment_score,
                        impact_level=self._assess_impact(news),
                        category=news.get('category', 'general')
                    )
                    
                    self.news_buffer.put(news_event, timeout=1)
                
                time.sleep(60)  # Check news every minute
                
            except Exception as e:
                logger.error(f"Error processing news data: {e}")
                self.metrics['errors'] += 1
                self.metrics['last_error'] = str(e)
                time.sleep(60)
    
    def _process_economic_data(self):
        """Process economic calendar data"""
        while self.is_running:
            try:
                # Get economic calendar events
                economic_events = self._fetch_economic_calendar()
                
                for event_data in economic_events:
                    economic_event = EconomicEvent(
                        timestamp=event_data['timestamp'],
                        country=event_data['country'],
                        event=event_data['event'],
                        actual=event_data.get('actual'),
                        forecast=event_data.get('forecast'),
                        previous=event_data.get('previous'),
                        impact=event_data.get('impact', 'low'),
                        currency=event_data.get('currency', 'USD')
                    )
                    
                    self.economic_buffer.put(economic_event, timeout=1)
                
                time.sleep(300)  # Check economic calendar every 5 minutes
                
            except Exception as e:
                logger.error(f"Error processing economic data: {e}")
                self.metrics['errors'] += 1
                self.metrics['last_error'] = str(e)
                time.sleep(300)
    
    def _process_social_data(self):
        """Process social media sentiment data"""
        while self.is_running:
            try:
                # Get social media data
                social_data = self._fetch_social_media()
                
                for post in social_data:
                    sentiment_score = self._analyze_sentiment(post['text'])
                    
                    social_event = MarketEvent(
                        timestamp=datetime.now(),
                        event_type='social',
                        symbol=post.get('symbol'),
                        data={
                            'text': post['text'],
                            'sentiment_score': sentiment_score,
                            'platform': post['platform'],
                            'author': post.get('author'),
                            'engagement': post.get('engagement', 0)
                        },
                        source=post['platform'],
                        confidence=0.7  # Lower confidence for social media
                    )
                    
                    self.social_buffer.put(social_event, timeout=1)
                
                time.sleep(30)  # Check social media every 30 seconds
                
            except Exception as e:
                logger.error(f"Error processing social data: {e}")
                self.metrics['errors'] += 1
                self.metrics['last_error'] = str(e)
                time.sleep(30)
    
    def _process_events(self):
        """Process all events from buffers"""
        while self.is_running:
            try:
                # Process price events
                self._process_buffer_events(self.price_buffer, 'price')
                
                # Process news events
                self._process_buffer_events(self.news_buffer, 'news')
                
                # Process economic events
                self._process_buffer_events(self.economic_buffer, 'economic')
                
                # Process social events
                self._process_buffer_events(self.social_buffer, 'social')
                
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
                
            except Exception as e:
                logger.error(f"Error processing events: {e}")
                self.metrics['errors'] += 1
                self.metrics['last_error'] = str(e)
                time.sleep(1)
    
    def _process_buffer_events(self, buffer: queue.Queue, event_type: str):
        """Process events from a specific buffer"""
        try:
            while not buffer.empty():
                event = buffer.get_nowait()
                
                # Calculate latency
                latency = (datetime.now() - event.timestamp).total_seconds()
                self.metrics['average_latency'] = (
                    self.metrics['average_latency'] * 0.9 + latency * 0.1
                )
                
                # Call event handlers
                for handler in self.event_handlers.get(event_type, []):
                    try:
                        handler(event)
                    except Exception as e:
                        logger.error(f"Error in event handler: {e}")
                
                self.metrics['events_processed'] += 1
                
        except queue.Empty:
            pass
    
    def _get_latest_prices(self) -> Dict[str, Dict[str, Any]]:
        """Get latest price data (simulated)"""
        # In practice, this would connect to real data feeds
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']
        prices = {}
        
        for symbol in symbols:
            # Simulate price movement
            base_price = 1.1000 if 'USD' in symbol else 1800.0
            price_change = np.random.normal(0, 0.001)
            current_price = base_price * (1 + price_change)
            
            prices[symbol] = {
                'price': current_price,
                'bid': current_price - 0.0001,
                'ask': current_price + 0.0001,
                'volume': np.random.randint(1000, 10000),
                'timestamp': datetime.now()
            }
        
        return prices
    
    def _fetch_news(self) -> List[Dict[str, Any]]:
        """Fetch latest news (simulated)"""
        # In practice, this would connect to news APIs
        news_items = [
            {
                'title': 'Fed Signals Potential Rate Cut',
                'content': 'The Federal Reserve indicated it may consider cutting interest rates...',
                'source': 'Reuters',
                'symbol': 'USD',
                'category': 'monetary_policy'
            },
            {
                'title': 'European Inflation Data Released',
                'content': 'Eurozone inflation came in higher than expected...',
                'source': 'Bloomberg',
                'symbol': 'EUR',
                'category': 'economic_data'
            }
        ]
        
        return news_items
    
    def _fetch_economic_calendar(self) -> List[Dict[str, Any]]:
        """Fetch economic calendar events (simulated)"""
        # In practice, this would connect to economic calendar APIs
        events = [
            {
                'timestamp': datetime.now() + timedelta(hours=2),
                'country': 'US',
                'event': 'Non-Farm Payrolls',
                'forecast': 200000,
                'previous': 195000,
                'impact': 'high',
                'currency': 'USD'
            },
            {
                'timestamp': datetime.now() + timedelta(hours=4),
                'country': 'EU',
                'event': 'GDP Growth Rate',
                'forecast': 0.3,
                'previous': 0.2,
                'impact': 'medium',
                'currency': 'EUR'
            }
        ]
        
        return events
    
    def _fetch_social_media(self) -> List[Dict[str, Any]]:
        """Fetch social media data (simulated)"""
        # In practice, this would connect to social media APIs
        posts = [
            {
                'text': 'EUR/USD looking bullish today! #forex #trading',
                'platform': 'Twitter',
                'symbol': 'EURUSD',
                'author': 'trader123',
                'engagement': 50
            },
            {
                'text': 'Gold prices surging on safe haven demand',
                'platform': 'Reddit',
                'symbol': 'XAUUSD',
                'author': 'goldbug',
                'engagement': 25
            }
        ]
        
        return posts
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text"""
        if not SENTIMENT_AVAILABLE:
            return 0.0
        
        try:
            blob = TextBlob(text)
            sentiment = blob.sentiment.polarity
            return sentiment
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return 0.0
    
    def _assess_impact(self, news: Dict[str, Any]) -> str:
        """Assess impact level of news"""
        high_impact_keywords = ['fed', 'rate', 'inflation', 'gdp', 'employment', 'crisis']
        medium_impact_keywords = ['earnings', 'trade', 'oil', 'gold', 'currency']
        
        text = (news['title'] + ' ' + news['content']).lower()
        
        if any(keyword in text for keyword in high_impact_keywords):
            return 'high'
        elif any(keyword in text for keyword in medium_impact_keywords):
            return 'medium'
        else:
            return 'low'
    
    def get_market_sentiment(self, symbol: str, timeframe: str = '1h') -> Dict[str, Any]:
        """Get aggregated market sentiment for a symbol"""
        # Aggregate sentiment from news and social media
        news_sentiment = self._get_news_sentiment(symbol, timeframe)
        social_sentiment = self._get_social_sentiment(symbol, timeframe)
        
        # Weighted average
        total_sentiment = (news_sentiment * 0.7 + social_sentiment * 0.3)
        
        return {
            'symbol': symbol,
            'overall_sentiment': total_sentiment,
            'news_sentiment': news_sentiment,
            'social_sentiment': social_sentiment,
            'confidence': 0.8,
            'timestamp': datetime.now()
        }
    
    def _get_news_sentiment(self, symbol: str, timeframe: str) -> float:
        """Get news sentiment for symbol"""
        # In practice, this would query the news buffer
        return np.random.uniform(-1, 1)  # Simulated sentiment
    
    def _get_social_sentiment(self, symbol: str, timeframe: str) -> float:
        """Get social media sentiment for symbol"""
        # In practice, this would query the social buffer
        return np.random.uniform(-1, 1)  # Simulated sentiment
    
    def get_economic_events(self, currency: str, hours_ahead: int = 24) -> List[EconomicEvent]:
        """Get upcoming economic events for a currency"""
        events = []
        cutoff_time = datetime.now() + timedelta(hours=hours_ahead)
        
        # In practice, this would query the economic buffer
        # For now, return simulated events
        return events
    
    def get_real_time_features(self, symbol: str) -> Dict[str, Any]:
        """Get real-time features for ML models"""
        # Get latest price data
        price_data = self._get_latest_prices().get(symbol, {})
        
        # Get sentiment data
        sentiment_data = self.get_market_sentiment(symbol)
        
        # Get economic events
        currency = symbol[:3] if len(symbol) >= 3 else 'USD'
        economic_events = self.get_economic_events(currency, 24)
        
        # Calculate features
        features = {
            'price': price_data.get('price', 0),
            'bid_ask_spread': price_data.get('ask', 0) - price_data.get('bid', 0),
            'volume': price_data.get('volume', 0),
            'sentiment_score': sentiment_data['overall_sentiment'],
            'news_sentiment': sentiment_data['news_sentiment'],
            'social_sentiment': sentiment_data['social_sentiment'],
            'upcoming_high_impact_events': len([e for e in economic_events if e.impact == 'high']),
            'upcoming_medium_impact_events': len([e for e in economic_events if e.impact == 'medium']),
            'timestamp': datetime.now()
        }
        
        return features
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get pipeline performance metrics"""
        return {
            'is_running': self.is_running,
            'events_processed': self.metrics['events_processed'],
            'average_latency': self.metrics['average_latency'],
            'errors': self.metrics['errors'],
            'last_error': self.metrics['last_error'],
            'buffer_sizes': {
                'price': self.price_buffer.qsize(),
                'news': self.news_buffer.qsize(),
                'economic': self.economic_buffer.qsize(),
                'social': self.social_buffer.qsize()
            },
            'last_update': self.last_update
        }
    
    def clear_buffers(self):
        """Clear all data buffers"""
        while not self.price_buffer.empty():
            try:
                self.price_buffer.get_nowait()
            except queue.Empty:
                break
        
        while not self.news_buffer.empty():
            try:
                self.news_buffer.get_nowait()
            except queue.Empty:
                break
        
        while not self.economic_buffer.empty():
            try:
                self.economic_buffer.get_nowait()
            except queue.Empty:
                break
        
        while not self.social_buffer.empty():
            try:
                self.social_buffer.get_nowait()
            except queue.Empty:
                break
        
        logger.info("All buffers cleared")
