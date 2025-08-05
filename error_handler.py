#!/usr/bin/env python3
"""
Error Handling and Monitoring System
===================================

Comprehensive error handling, monitoring, and alerting system for the trading bot.
Includes custom exceptions, error tracking, and performance monitoring.
"""

import logging
import traceback
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from functools import wraps

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class ErrorCategory(Enum):
    """Error categories"""
    CONNECTION = "CONNECTION"
    TRADING = "TRADING"
    ANALYSIS = "ANALYSIS"
    DATA = "DATA"
    CONFIGURATION = "CONFIGURATION"
    SYSTEM = "SYSTEM"

@dataclass
class ErrorRecord:
    """Error record for tracking"""
    timestamp: datetime
    error_type: str
    message: str
    severity: ErrorSeverity
    category: ErrorCategory
    stack_trace: str
    context: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    resolution_notes: Optional[str] = None

@dataclass
class PerformanceMetric:
    """Performance metric for monitoring"""
    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    context: Dict[str, Any] = field(default_factory=dict)

class TradingBotError(Exception):
    """Base exception for trading bot errors"""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 category: ErrorCategory = ErrorCategory.SYSTEM, context: Dict[str, Any] = None):
        super().__init__(message)
        self.severity = severity
        self.category = category
        self.context = context or {}
        self.timestamp = datetime.now()

class ConnectionError(TradingBotError):
    """MT5 connection related errors"""
    
    def __init__(self, message: str, context: Dict[str, Any] = None):
        super().__init__(message, ErrorSeverity.HIGH, ErrorCategory.CONNECTION, context)

class TradingError(TradingBotError):
    """Trading execution related errors"""
    
    def __init__(self, message: str, context: Dict[str, Any] = None):
        super().__init__(message, ErrorSeverity.CRITICAL, ErrorCategory.TRADING, context)

class AnalysisError(TradingBotError):
    """Market analysis related errors"""
    
    def __init__(self, message: str, context: Dict[str, Any] = None):
        super().__init__(message, ErrorSeverity.MEDIUM, ErrorCategory.ANALYSIS, context)

class DataError(TradingBotError):
    """Data retrieval and processing errors"""
    
    def __init__(self, message: str, context: Dict[str, Any] = None):
        super().__init__(message, ErrorSeverity.MEDIUM, ErrorCategory.DATA, context)

class ConfigurationError(TradingBotError):
    """Configuration related errors"""
    
    def __init__(self, message: str, context: Dict[str, Any] = None):
        super().__init__(message, ErrorSeverity.HIGH, ErrorCategory.CONFIGURATION, context)

class ErrorHandler:
    """Centralized error handling and monitoring"""
    
    def __init__(self, log_file: str = "logs/errors.log"):
        self.logger = logging.getLogger('trading_bot.errors')
        self.error_records: List[ErrorRecord] = []
        self.performance_metrics: List[PerformanceMetric] = []
        self.error_callbacks: List[Callable] = []
        self.performance_callbacks: List[Callable] = []
        
        # Create logs directory
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Setup file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.ERROR)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> ErrorRecord:
        """Handle and record an error"""
        context = context or {}
        
        # Determine error type and severity
        if isinstance(error, TradingBotError):
            error_type = type(error).__name__
            severity = error.severity
            category = error.category
            message = str(error)
            context.update(error.context)
        else:
            error_type = type(error).__name__
            severity = ErrorSeverity.MEDIUM
            category = ErrorCategory.SYSTEM
            message = str(error)
        
        # Create error record
        error_record = ErrorRecord(
            timestamp=datetime.now(),
            error_type=error_type,
            message=message,
            severity=severity,
            category=category,
            stack_trace=traceback.format_exc(),
            context=context
        )
        
        # Add to records
        self.error_records.append(error_record)
        
        # Log error
        log_message = f"{error_type}: {message} | Severity: {severity.value} | Category: {category.value}"
        if context:
            log_message += f" | Context: {json.dumps(context)}"
        
        if severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
        
        # Call error callbacks
        for callback in self.error_callbacks:
            try:
                callback(error_record)
            except Exception as callback_error:
                self.logger.error(f"Error in error callback: {callback_error}")
        
        return error_record
    
    def record_performance_metric(self, metric_name: str, value: float, unit: str, 
                                 context: Dict[str, Any] = None) -> PerformanceMetric:
        """Record a performance metric"""
        context = context or {}
        
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_name=metric_name,
            value=value,
            unit=unit,
            context=context
        )
        
        self.performance_metrics.append(metric)
        
        # Call performance callbacks
        for callback in self.performance_callbacks:
            try:
                callback(metric)
            except Exception as callback_error:
                self.logger.error(f"Error in performance callback: {callback_error}")
        
        return metric
    
    def get_recent_errors(self, hours: int = 24, severity: Optional[ErrorSeverity] = None) -> List[ErrorRecord]:
        """Get recent errors within specified time range"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_errors = [e for e in self.error_records if e.timestamp >= cutoff_time]
        
        if severity:
            recent_errors = [e for e in recent_errors if e.severity == severity]
        
        return recent_errors
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary statistics"""
        recent_errors = self.get_recent_errors(hours)
        
        summary = {
            'total_errors': len(recent_errors),
            'by_severity': {},
            'by_category': {},
            'most_common_errors': {},
            'unresolved_errors': len([e for e in recent_errors if not e.resolved])
        }
        
        # Count by severity
        for severity in ErrorSeverity:
            summary['by_severity'][severity.value] = len([e for e in recent_errors if e.severity == severity])
        
        # Count by category
        for category in ErrorCategory:
            summary['by_category'][category.value] = len([e for e in recent_errors if e.category == category])
        
        # Most common errors
        error_types = {}
        for error in recent_errors:
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
        
        summary['most_common_errors'] = dict(sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5])
        
        return summary
    
    def resolve_error(self, error_record: ErrorRecord, notes: str = None):
        """Mark an error as resolved"""
        error_record.resolved = True
        error_record.resolution_time = datetime.now()
        error_record.resolution_notes = notes
    
    def add_error_callback(self, callback: Callable[[ErrorRecord], None]):
        """Add error callback function"""
        self.error_callbacks.append(callback)
    
    def add_performance_callback(self, callback: Callable[[PerformanceMetric], None]):
        """Add performance callback function"""
        self.performance_callbacks.append(callback)
    
    def cleanup_old_records(self, days: int = 30):
        """Clean up old error and performance records"""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        # Clean up error records
        self.error_records = [e for e in self.error_records if e.timestamp >= cutoff_time]
        
        # Clean up performance metrics
        self.performance_metrics = [p for p in self.performance_metrics if p.timestamp >= cutoff_time]

def handle_errors(error_handler: ErrorHandler, context: Dict[str, Any] = None):
    """Decorator for error handling"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler.handle_error(e, context or {})
                raise
        return wrapper
    return decorator

def monitor_performance(error_handler: ErrorHandler, metric_name: str, unit: str = "ms"):
    """Decorator for performance monitoring"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                error_handler.record_performance_metric(
                    metric_name, execution_time, unit,
                    {'function': func.__name__, 'success': True}
                )
                return result
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                error_handler.record_performance_metric(
                    metric_name, execution_time, unit,
                    {'function': func.__name__, 'success': False, 'error': str(e)}
                )
                raise
        return wrapper
    return decorator

class AlertManager:
    """Alert management system"""
    
    def __init__(self, error_handler: ErrorHandler):
        self.error_handler = error_handler
        self.alert_rules: List[Dict] = []
        self.alert_history: List[Dict] = []
    
    def add_alert_rule(self, rule: Dict):
        """Add an alert rule"""
        self.alert_rules.append(rule)
    
    def check_alerts(self):
        """Check and trigger alerts based on rules"""
        for rule in self.alert_rules:
            if self._should_trigger_alert(rule):
                self._trigger_alert(rule)
    
    def _should_trigger_alert(self, rule: Dict) -> bool:
        """Check if alert should be triggered"""
        # Implementation depends on alert rule structure
        # This is a placeholder for alert logic
        return False
    
    def _trigger_alert(self, rule: Dict):
        """Trigger an alert"""
        alert = {
            'timestamp': datetime.now(),
            'rule': rule,
            'message': rule.get('message', 'Alert triggered')
        }
        self.alert_history.append(alert)
        # Here you would implement actual alert sending (email, SMS, etc.)

# Global error handler instance
error_handler = ErrorHandler()
alert_manager = AlertManager(error_handler) 