"""
Data Sources Package
Handles external data fetching from APIs and databases
"""

from .external_data_fetcher import (
    IMDWeatherFetcher,
    FertilizerDataFetcher,
    ResearchPaperFetcher,
    LocationDataFetcher,
    ExternalDataIntegrator
)

__all__ = [
    'IMDWeatherFetcher',
    'FertilizerDataFetcher',
    'ResearchPaperFetcher',
    'LocationDataFetcher',
    'ExternalDataIntegrator'
]
