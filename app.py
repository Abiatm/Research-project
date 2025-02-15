from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import pickle
import pandas as pd 
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(debug=True)

# Add CORS middleware (if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
try:
    with open("Reseach_project_model.pkl", "rb") as model_file:
        loaded_model = pickle.load(model_file)
    app.state.model_loaded = True 
except FileNotFoundError:
    app.state.model_loaded = False 
    app.state.error_message = "Model file 'Ml_project_model.pkl' not found."
except Exception as e:
    app.state.model_loaded = False 
    app.state.error_message = f"Error loading model: {e}"

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
    if not app.state.model_loaded:
        return JSONResponse({"error": app.state.error_message}, status_code=500)

    try:
        # Create a DataFrame from the input dictionary
        df = pd.DataFrame([data], columns=['ground', 'red', 'yellow', 'blue', 'Ir', 'Iy', 'Ib', 'Vr', 'Vy', 'Vb'])

        prediction = loaded_model.predict(df)[0] 

        # Convert prediction to "healthy" or "unhealthy"
        health_status = "healthy" if prediction == 0 else "unhealthy"

        return JSONResponse({"prediction": health_status})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 


    
