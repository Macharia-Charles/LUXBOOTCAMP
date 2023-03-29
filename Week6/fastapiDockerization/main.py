from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/users")
async def users():
    users = [
        {
            "name": "Charles Macharia",
            "age": 24,
            "city": "Nanyuki, Kenya"
        },

        {
            "name": "Polycarp King'ori",
            "age": 23,
            "city": "Nyahururu, Kenya"
        },

         {
            "name": "Brian Omollo",
            "age": 25,
            "city": "Homabay, Kenya"
        }
    ]

    return users
