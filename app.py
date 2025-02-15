from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import pickle
import pandas as pd

app = FastAPI(debug=True)

# Load the trained model
try:
    model = pickle.load(open("Reseach_project_model.pkl", "rb")) 
except FileNotFoundError:
    return JSONResponse({"error": "Model file not found."}, status_code=500)
except Exception as e:
    return JSONResponse({"error": f"Error loading model: {e}"}, status_code=500)

@app.get("/")
def home():
    return {"message": "Welcome to the prediction service"}

@app.post("/predict")
async def predict(data: dict):
    """
    Predicts the health status based on input data.

    Args:
        data: A dictionary containing the input features: 
              {'ground': str, 'red': str, 'yellow': str, 'blue': str, 
               'Ir': str, 'Iy': str, 'Ib': str, 'Vr': str, 'Vy': str, 'Vb': str}

    Returns:
        A JSONResponse with the predicted health status ("healthy" or "unhealthy"). 
    """
    try:
        # Create a DataFrame from the input dictionary
        df = pd.DataFrame([data], columns=['ground', 'red', 'yellow', 'blue', 'Ir', 'Iy', 'Ib', 'Vr', 'Vy', 'Vb'])

        prediction = model.predict(df)[0]  # Get the prediction directly

        # Convert prediction to "healthy" or "unhealthy"
        health_status = "healthy" if prediction == 0 else "unhealthy"

        return JSONResponse({"prediction": health_status})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
