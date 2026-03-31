from pydantic import BaseModel , Field



# model means a class that defines the structure of data we want to work with.
# so we can give types

# we have inherit BaseModel to create our own model

class UserModel(BaseModel):
    # ... means that this field is required
    id: int = Field(..., description="User ID")
    email: str = Field(..., description="User email")
    name: str = Field(..., description="User name")
    
u1 = UserModel(id=1, email="Oxh9I@example.com", name="John Doe")

print(u1)