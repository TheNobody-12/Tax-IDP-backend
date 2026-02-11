import csv
from typing import Optional, Union, List
from src.extraction.models import MedicalExpense

def _parse_page_numbers(page_numbers: Union[str, List[int]]) -> list[int]:
    """Parse string or list of page numbers into list of ints"""
    if isinstance(page_numbers, list):
        return sorted(set(page_numbers))
        
    pages = []
    for part in page_numbers.split(","):
        part = part.strip()
        if not part: continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            try:
                start = int(start_str.strip())
                end = int(end_str.strip())
                pages.extend(list(range(start, end + 1)))
            except ValueError:
                continue
        else:
            try:
                pages.append(int(part))
            except ValueError:
                continue
    return sorted(set(pages))

def _format_page_numbers(pages: list[int]) -> str:
    """Format list of ints into page range string"""
    if not pages:
        return ""
    pages = sorted(set(pages))
    ranges = []
    start = prev = pages[0]
    for p in pages[1:]:
        if p == prev + 1:
            prev = p
            continue
        ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = p
    ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ",".join(ranges)

def normalize_expense_payload(exp: dict) -> dict:
    """Normalize fields in the raw dictionary before Pydantic conversion"""
    amount = exp.get("amount")
    if amount == "" or amount is None:
        exp["amount"] = None
    
    eligible = exp.get("eligible")
    if isinstance(eligible, str):
        normalized = eligible.strip().lower()
        if normalized in {"true", "yes", "y", "eligible"}:
            exp["eligible"] = True
        elif normalized in {"false", "no", "n", "not eligible", "ineligible", "unclaimable"}:
            exp["eligible"] = False
        else:
            exp["eligible"] = None
    return exp

def post_process_expenses(expenses: list[MedicalExpense]) -> list[MedicalExpense]:
    """Merge duplicates and clean up comments"""
    merged: dict[tuple, MedicalExpense] = {}
    page_map: dict[tuple, list[int]] = {}
    
    for exp in expenses:
        key = (
            exp.date or "",
            exp.amount,
            exp.payee_provider or "",
            exp.rx_invoice_num or "",
            exp.expense_type or "",
        )
        pages = _parse_page_numbers(exp.page_numbers or [])
        
        if key not in merged:
            merged[key] = exp
            page_map[key] = pages
            continue
            
        page_map[key].extend(pages)
        # We'll update the list in MedicalExpense
        merged[key].page_numbers = sorted(set(page_map[key]))
        
        if merged[key].comments and exp.comments:
            merged[key].comments = f"{merged[key].comments}; {exp.comments}"
        elif exp.comments:
            merged[key].comments = exp.comments

    finalized = []
    for exp in merged.values():
        if exp.eligible is False and not exp.comments:
            exp.comments = "Not claimable"
        if exp.amount is None and not exp.comments:
            exp.comments = "Unknown amount"
        finalized.append(exp)

    def sort_key(item: MedicalExpense) -> int:
        pages = item.page_numbers
        # Ensure we have a list of ints
        if not pages:
            return 0
        try:
            # Handle potential string "1" or list ["1"] or list [1]
            if isinstance(pages, list):
                first = pages[0]
            else:
                first = pages
                
            return int(first)
        except (ValueError, TypeError):
            return 0

    return sorted(finalized, key=sort_key)

def save_to_csv(expenses: list[MedicalExpense], output_path: str) -> None:
    """Save the list of MedicalExpense objects to a CSV file"""
    headers = [
        "Page #",
        "Date",
        "Amount ($)",
        "Eligible",
        "Payee/Provider",
        "Rx # / Invoice #",
        "Type of Medical Expense",
        "Duplicate Reference",
        "Comments/Notes"
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for expense in expenses:
            eligible_str = "Yes" if expense.eligible else "No" if expense.eligible is False else ""
            page_str = _format_page_numbers(expense.page_numbers)
            writer.writerow([
                page_str,
                expense.date,
                f"{expense.amount:.2f}" if expense.amount is not None else "",
                eligible_str,
                expense.payee_provider,
                expense.rx_invoice_num or "",
                expense.expense_type,
                expense.duplicate_reference or "",
                expense.comments or ""
            ])
