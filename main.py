from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# 加载模型
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("spam_classifier.pkl")

app = FastAPI(
    title="SMS Spam Detection API",
    description="A microservice that classifies SMS messages as spam or ham.",
    version="1.0"
)

# 定义输入格式
class MessageInput(BaseModel):
    text: str

# 首页
@app.get("/")
def home():
    return {
        "message": "SMS Spam Detection API is running.",
        "usage": "Go to /docs to test the API."
    }

# 健康检查
@app.get("/health")
def health():
    return {"status": "ok"}

# 预测接口
@app.post("/predict")
def predict(input_data: MessageInput):
    text = input_data.text.strip()

    if text == "":
        return {"error": "Input text cannot be empty."}

    vectorized = vectorizer.transform([text])
    prediction = model.predict(vectorized)[0]

    return {
        "input_text": text,
        "prediction": prediction
    }