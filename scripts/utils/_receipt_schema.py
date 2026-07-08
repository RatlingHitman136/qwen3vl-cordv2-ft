# receipt_schema.py
import json
from typing import List, Optional
from pydantic import BaseModel, ConfigDict

class SubMenuItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    nm: Optional[str] = None         # 404/0
    cnt: Optional[str] = None        # 191/0
    price: Optional[str] = None      # 157/0
    unitprice: Optional[str] = None  # 14/0

class MenuItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    nm: Optional[List[str]] = None             # mixed (2565 str / 4 list)
    price: Optional[str] = None                # pure str (2559/0)
    cnt: Optional[List[str]] = None            # mixed (2330/1)
    unitprice: Optional[str] = None            # pure str (737/0)
    num: Optional[List[str]] = None            # mixed (107/1)
    discountprice: Optional[List[str]] = None  # mixed (93/4)
    etc: Optional[str] = None                  # pure str (9/0)
    itemsubtotal: Optional[str] = None         # pure str (7/0)
    sub: Optional[List[SubMenuItem]] = None    # CONFIRMED list via container coercion

class Total(BaseModel):
    model_config = ConfigDict(extra="forbid")
    total_price: Optional[List[str]] = None     # mixed
    cashprice: Optional[List[str]] = None       # mixed
    changeprice: Optional[List[str]] = None     # mixed
    creditcardprice: Optional[List[str]] = None # mixed
    menuqty_cnt: Optional[str] = None
    menutype_cnt: Optional[str] = None
    emoneyprice: Optional[str] = None
    total_etc: Optional[str] = None

class SubTotal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subtotal_price: Optional[List[str]] = None  # mixed
    tax_price: Optional[List[str]] = None       # mixed
    service_price: Optional[str] = None
    etc: Optional[List[str]] = None             # mixed
    discount_price: Optional[List[str]] = None  # mixed

class Receipt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    menu: Optional[List[MenuItem]] = None
    total: Optional[Total] = None
    sub_total: Optional[SubTotal] = None

def assert_schema_valid(split, name):
    # LazySplit exposes .labels — validate those directly so this never
    # triggers per-row image decoding; plain lists of rows still work
    labels = getattr(split, "labels", None)
    if labels is None:
        labels = [ex["label"] for ex in split]
    bad = 0
    for label in labels:
        try:
            Receipt.model_validate(label)
        except Exception as e:
            bad += 1
            if bad <= 3:
                print(f"[{name}] reject: {e}\n  label={json.dumps(label)[:200]}")
    print(f"[{name}] {len(split)-bad}/{len(split)} valid")
    return bad