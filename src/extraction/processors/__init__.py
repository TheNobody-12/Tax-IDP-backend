"""Processors package for document processing pipeline."""

from src.extraction.base import PROCESSOR_REGISTRY
from src.extraction.classifier import DocumentCategory
from src.extraction.models import (
    TaxSlip, Donation, ChildcareExpense, RRSPContribution,
    FHSAContribution, PropertyTax, RentReceipt, UnionDue,
    OtherDocument
)
from .medical_expense import MedicalExpenseProcessor
from .generic import GenericExpenseProcessor

# Register medical expense processor
PROCESSOR_REGISTRY.register(MedicalExpenseProcessor())

# Register category-specific instances of GenericExpenseProcessor
PROCESSOR_REGISTRY.register(GenericExpenseProcessor(
    category=DocumentCategory.SLIPS,
    item_model=TaxSlip,
    output_name="slips"
))

PROCESSOR_REGISTRY.register(GenericExpenseProcessor(
    category=DocumentCategory.CHARITABLE_DONATIONS,
    item_model=Donation,
    output_name="donation"
))

PROCESSOR_REGISTRY.register(GenericExpenseProcessor(
    category=DocumentCategory.POLITICAL_DONATIONS,
    item_model=Donation,
    output_name="donation"
))

PROCESSOR_REGISTRY.register(GenericExpenseProcessor(
    category=DocumentCategory.CHILD_CARE_EXPENSES,
    item_model=ChildcareExpense,
    output_name="child_care"
))

PROCESSOR_REGISTRY.register(GenericExpenseProcessor(
    category=DocumentCategory.RRSP_CONTRIBUTION,
    item_model=RRSPContribution,
    output_name="rrsp_contribution"
))

PROCESSOR_REGISTRY.register(GenericExpenseProcessor(
    category=DocumentCategory.FHSA_CONTRIBUTION,
    item_model=FHSAContribution,
    output_name="fhsa_contribution"
))

PROCESSOR_REGISTRY.register(GenericExpenseProcessor(
    category=DocumentCategory.UNION_PROFESSIONAL_DUES,
    item_model=UnionDue,
    output_name="union_dues"
))

PROCESSOR_REGISTRY.register(GenericExpenseProcessor(
    category=DocumentCategory.PROPERTY_TAX_RECEIPT,
    item_model=PropertyTax,
    output_name="property_tax"
))

PROCESSOR_REGISTRY.register(GenericExpenseProcessor(
    category=DocumentCategory.RENT_RECEIPT,
    item_model=RentReceipt,
    output_name="rent_receipt"
))

PROCESSOR_REGISTRY.register(GenericExpenseProcessor(
    category=DocumentCategory.OTHER_DOCUMENTS,
    item_model=OtherDocument,
    output_name="other_docs"
))

__all__ = ["MedicalExpenseProcessor", "GenericExpenseProcessor"]
