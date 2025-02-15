from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import pickle
import gunicorn

app = FastAPI(debug=True)

@app.get("/")
def home():
    return {"text": "Prediction result"}

@app.post("/predict")
def predict(ground: str, red: str, yellow: str, blue: str, Ir: str, Iy: str, Ib: str, Vr: str, Vy: str, Vb: str):
    try:
        print("Input parameters:", ground, red, yellow, blue, Ir, Iy, Ib, Vr, Vy, Vb)  # Debug print
        model = pickle.load(open("Reseach_project_model.pkl", "rb"))

        # Convert input parameters to the correct data types if necessary
        ground = int(ground)  # Example: Convert to integer
        red = int(red)
        yellow = int(yellow)
        blue = int(blue)
        Ir = float(Ir)  # Example: Convert to float
        Iy = float(Iy)
        Ib = float(Ib)
        Vr = float(Vr)
        Vy = float(Vy)
        Vb = float(Vb)


        makeprediction = model.predict([[ground, red, yellow, blue, Ir, Iy, Ib, Vr, Vy, Vb]])
        print("Prediction result:", makeprediction)  # Debug print

        if makeprediction[0] == 0:
            result = "healthy"
        else:
            result = "unhealthy"

        return {"prediction": result}
    except Exception as e:
        print("Error during prediction:", e)  # Print error message
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)  # Changed for Render deployment
