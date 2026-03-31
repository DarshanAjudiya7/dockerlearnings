from pydantic import BaseModel




class ResponseModel(BaseModel):
    message: str
    data: dict



r1 = ResponseModel(message="Hello, World!", data={"key": "value"})
print(r1)
