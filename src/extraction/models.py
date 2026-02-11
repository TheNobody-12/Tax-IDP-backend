from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict
import re

# ============================================================================
# Shared Components
# ============================================================================

class ClientInfo(BaseModel):
    """Structured client information"""
    model_config = ConfigDict(extra="forbid")
    
    name: Optional[str] = Field(default="", description="Client full name")
    address: Optional[str] = Field(default="", description="Client address")
    sin: Optional[str] = Field(default="", description="Social Insurance Number")
    tax_id: Optional[str] = Field(default="", description="Other tax identifier")

# ============================================================================
# Document Field Models
# ============================================================================

class BaseExpense(BaseModel):
    """Base model for all expense/income types"""
    model_config = ConfigDict(extra="forbid")
    
    page_numbers: List[int] = Field(default_factory=list, description="Page numbers (e.g., '1-2', '5')")
    comments: Optional[str] = Field(default="", description="Notes or observations")

    @field_validator('*', mode='before')
    def clean_strings(cls, v):
        if isinstance(v, str):
            cleaned = v.strip()
            return cleaned if cleaned else None
        return v
    
    @field_validator('amount', check_fields=False)
    def clean_amount(cls, v):
        if isinstance(v, (float, int)):
            return float(v)
        if isinstance(v, str):
            # Remove currency symbols and commas
            clean = re.sub(r'[$,CADc\s]', '', v)
            try:
                return float(clean)
            except ValueError:
                return None
        return v

class ChildcareExpense(BaseExpense):
    """1) Childcare"""
    expense_date: Optional[str] = Field(default=None, description="YYYY-MM-DD or null")
    child_name: Optional[str] = Field(default=None)
    provider_name: Optional[str] = Field(default=None)
    city: Optional[str] = Field(default=None)
    description: Optional[str] = Field(default=None)
    amount: Optional[float] = Field(default=None)

class Donation(BaseExpense):
    """2) Donation"""
    donation_date: Optional[str] = Field(default=None, description="YYYY-MM-DD or null")
    donor_name: Optional[str] = Field(default=None)
    donee_name: Optional[str] = Field(default=None, description="Charity name")
    amount: Optional[float] = Field(default=None)

class FHSAContribution(BaseExpense):
    """3) FHSAContribution"""
    sin: Optional[str] = Field(default=None)
    amount: Optional[float] = Field(default=None)

class TaxSlip(BaseExpense):
    """4) InvestmentSlip / FactSlipBox / T-Slips"""
    sin: Optional[str] = Field(default=None)
    slip_type: str = Field(description="Slip Type (e.g. T4, T5, T3, T5008)")
    issuer: Optional[str] = Field(default=None, description="Issuer or Employer Name")
    
    # Detailed box values
    box_values: Dict[str, Optional[Union[float, str]]] = Field(default_factory=dict, description="Map of box numbers to amounts")
    
    # Summary fields (when available/applicable)
    amount: Optional[float] = Field(default=None, description="Primary income amount if applicable")
    currency: Optional[str] = Field(default="CAD")
    country: Optional[str] = Field(default="CAN")
    admin_fee: Optional[float] = Field(default=None)
    mode_of_holding: Optional[str] = Field(default=None)
    source_field_name: Optional[str] = Field(default=None)

class MedicalExpense(BaseExpense):
    """5) MedicalExpense"""
    date: Optional[str] = Field(default="", description="Transaction date in DD-MM-YYYY or MM-YYYY format")
    amount: Optional[float] = Field(default=None, description="Amount in dollars")
    payee_provider: Optional[str] = Field(default="", description="Medical provider or payee name")
    rx_invoice_num: Optional[str] = Field(default="", description="Rx number or invoice number")
    expense_type: Optional[str] = Field(default="", description="Standardized medical expense type")
    duplicate_reference: Optional[str] = Field(default="", description="Reference to duplicate pages if applicable")
    comments: Optional[str] = Field(default="", description="Notes, eligibility, or special remarks")
    eligible: Optional[bool] = Field(default=None, description="Whether expense is eligible for medical tax credit")

class PropertyTax(BaseExpense):
    """6) PropertyTax"""
    property_address: Optional[str] = Field(default=None)
    municipality: Optional[str] = Field(default=None)
    roll_number: Optional[str] = Field(default=None)
    tax_year_of_bill: Optional[int] = Field(default=None)
    amount: Optional[float] = Field(default=None)
    is_principal_residence: Optional[bool] = Field(default=None)

class RentReceipt(BaseExpense):
    """7) Rent"""
    property_address: Optional[str] = Field(default=None)
    landlord_name: Optional[str] = Field(default=None)
    months_paid: Optional[int] = Field(default=None)
    monthly_rent: Optional[float] = Field(default=None)
    total_rent: Optional[float] = Field(default=None, alias="amount")
    amount: Optional[float] = Field(default=None) # Alias for total_rent handling

class RRSPContribution(BaseExpense):
    """8) RRSPContribution"""
    sin: Optional[str] = Field(default=None)
    first_60_days_amount: Optional[float] = Field(default=None)
    rest_of_year_amount: Optional[float] = Field(default=None)
    is_spousal: Optional[bool] = Field(default=None)
    amount: Optional[float] = Field(default=None, description="Total amount if split not available")

class UnionDue(BaseExpense):
    """9) UnionDues"""
    organization_name: Optional[str] = Field(default=None)
    city: Optional[str] = Field(default=None)
    description: Optional[str] = Field(default=None)
    amount: Optional[float] = Field(default=None)

class OtherDocument(BaseExpense):
    """Fallback for unrecognized documents"""
    date: Optional[str] = Field(default=None, description="Relevant date found")
    description: Optional[str] = Field(default=None, description="Brief description of the document or item")
    amount: Optional[float] = Field(default=None, description="Any financial amount found")
    entity: Optional[str] = Field(default=None, description="Organization or person involved")
    category_guess: Optional[str] = Field(default=None, description="Guess at what this document represents")
    box_values: Dict[str, Optional[Union[float, str]]] = Field(default_factory=dict, description="Map of box numbers to amounts if this looks like a tax slip")

# ============================================================================
# Response Wrapper
# ============================================================================

class StructuredExtraction(BaseModel):
    """Top-level structure for extraction response"""
    model_config = ConfigDict(extra="forbid")
    
    client_info: Optional[ClientInfo]
    expenses: List[Union[
        MedicalExpense, 
        ChildcareExpense, 
        Donation, 
        FHSAContribution, 
        TaxSlip, 
        PropertyTax, 
        RentReceipt, 
        RRSPContribution, 
        UnionDue,
        OtherDocument
    ]] = Field(default_factory=list)
