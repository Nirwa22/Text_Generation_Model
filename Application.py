import os
from flask import Flask, request
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
Application = Flask(__name__)
CORS(Application)

load_dotenv()
API = os.getenv("SECRET_API_KEY")
model = AutoModelForCausalLM.from_pretrained("Nirwa22/Fine_tuned_model_final")
Tokenizer = AutoTokenizer.from_pretrained("Nirwa22/Fine_tuned_model_final")

@Application.route("/")
def home_route():
    return "Welcome"


@Application.route("/Enter_Data", methods=['POST'])
def enter_data():
    api = request.headers.get("Authorization")
    if api == API:
        try:
            data = request.get_json()
            if data["text"]:
                x = Tokenizer(data["text"], return_tensors="pt")
                Output = model.generate(x["input_ids"], max_length=50, do_sample=True, top_k=50, temperature=2.0,
                                        length_penalty=0.5, repetition_penalty=1.0)
                return {"Output": Tokenizer.decode(Output[0])}
            else:
                return {"Message": "Text_needed"}
        except Exception as e:
            return {"Message": e}
    elif not api:
        return {"Message": "API_Key needed"}
    elif api and api != API:
        return {"Message": "Unauthorized Access"}


if __name__ == "__main__":
    Application.run(debug = True)