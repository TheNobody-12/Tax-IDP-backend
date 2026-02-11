"""
Base processor interface for document processing pipeline.
All document processors (medical expenses, income, etc.) implement this interface.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProcessorResult:
    """Result from a processor"""
    processor_name: str
    items_extracted: int
    output_file: Optional[str]
    data: List[Dict[str, Any]]
    client_info: Optional[Dict[str, Any]] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


@dataclass
class DocumentContext:
    """Shared context for document processing"""
    doc_id: str
    pdf_path: str
    markdown_path: Optional[str]
    markdown_content: str
    total_pages: int
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseProcessor(ABC):
    """Abstract base class for document processors"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this processor"""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name (Category)"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description"""
        pass
    
    @property
    @abstractmethod
    def output_format(self) -> str:
        """Output format (csv, json, etc.)"""
        pass
    
    @abstractmethod
    async def process(self, context: DocumentContext, output_dir: str, ai_client: Optional[Any] = None) -> ProcessorResult:
        """
        Process the document and return results.
        
        Args:
            context: Document context with markdown content
            output_dir: Directory to write output files
            
        Returns:
            ProcessorResult with extracted data
        """
        pass
    
    def validate_context(self, context: DocumentContext) -> bool:
        """Validate that context is suitable for this processor"""
        return bool(context.markdown_content)


class ProcessorRegistry:
    """Registry of available processors"""
    
    def __init__(self):
        self._processors: Dict[str, BaseProcessor] = {}
    
    def register(self, processor: BaseProcessor) -> None:
        """Register a processor"""
        self._processors[processor.name] = processor
    
    def get(self, name: str) -> Optional[BaseProcessor]:
        """Get a processor by name"""
        return self._processors.get(name)
    
    def list_processors(self) -> List[str]:
        """List all registered processor names"""
        return list(self._processors.keys())
    
    def get_all(self) -> List[BaseProcessor]:
        """Get all registered processors"""
        return list(self._processors.values())


# Global registry
PROCESSOR_REGISTRY = ProcessorRegistry()


def register_processor(processor: BaseProcessor) -> BaseProcessor:
    """Decorator/function to register a processor"""
    PROCESSOR_REGISTRY.register(processor)
    return processor
