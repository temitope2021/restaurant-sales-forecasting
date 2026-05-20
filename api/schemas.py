from pydantic import BaseModel

class Features(BaseModel):
    sales_lag_7: float
    weekday: int  # must be int, not str
    is_holiday: bool