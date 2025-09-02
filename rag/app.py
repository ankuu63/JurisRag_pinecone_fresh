from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse    #for rendering html page 
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles     #for adding static i.e csss files 
from rag.pipeline import RAGPipeline
import shutil
import os
import traceback
# Initialize FastAPI application
app = FastAPI()


app.mount("/static", StaticFiles(directory="frontend"), name="static")


# Initialize the pipeline and igniting self
pipeline = RAGPipeline()

# API endpoints

@app.get("/")
def home():
    return FileResponse("frontend/index.html")



@app.post("/chat")
async def chat(query: str = Form(...), pdf: UploadFile = None):
   

    pdf_path = None  # Temporary storage path

    # If a PDF is uploaded,saving pdf to tmpfile
    if pdf:
        pdf_path = f"temp_{pdf.filename}"
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(pdf.file, buffer)

    try:
        # invoking pipeline with query and pdf path
        answer = pipeline.run(query=query, pdf_path=pdf_path)

       
        # FastAPI automatically converts Python dict -> JSON response
      
        return {"answer": str(answer)}


    #error monitoring to debug
    except Exception as e:
        print("Error in /chatendpoint")
        traceback.print_exc()

        return JSONResponse(content={"error": str(e)},
                            status_code=500
                            )
    

    finally:
        # Removing temp file
        if pdf_path and os.path.exists(pdf_path):
            os.remove(pdf_path)