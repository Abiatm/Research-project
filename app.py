from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import pickle

app = FastAPI(debug=True)

@app.get("/")
def home():
    return {"text": "Prediction result"}

@app.post("/predict")
def predict(ground: str, red: str, yellow: str, blue: str, Ir: str, Iy: str, Ib: str, Vr: str, Vy: str, Vb: str):
    try:
        model = pickle.load(open("Reseach_project_model.pkl", "rb"))
        makeprediction = model.predict([[ground, red, yellow, blue, Ir, Iy, Ib, Vr, Vy, Vb]])

        if makeprediction[0] == 0:
            result = "healthy"
        else:
            result = "unhealthy"

        return {"prediction": result}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)  # Changed for Render deployment
