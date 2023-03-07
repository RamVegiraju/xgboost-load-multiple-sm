import joblib
import os
import json
import pickle
import sagemaker_xgboost_container.encoder as xgb_encoders

"""
Deserialize fitted model
"""
def model_fn(model_dir):
    print("INSIDE THE MODEL FUNCTION--------------------------------")
    print(model_dir)
    model_list = os.listdir(model_dir)
    print("ALL MODELS WE ARE LOADING FROM TARBALL----------")
    print(model_list)
    model_hash = {}
    for model in model_list:
        if 'xgboost' not in model:
            continue
        else:
            loaded_model = pickle.load(open(os.path.join(model_dir, model), "rb"))
        model_hash[model] = loaded_model
    print("LISTING THE MAPPED MODEL DICTIONARY------------")
    print(model_hash)
    return model_hash


def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        request_body = json.loads(request_body)
        inpPayload = request_body['input']
        inpModels = request_body['models']
        return [inpPayload, inpModels]
    else:
        raise ValueError("This model only supports application/json input")

def predict_fn(input_data, model):
    print("INSIDE THE PREDICT FUNCTION---------------")
    print(input_data)
    print(model)
    output = []
    payload = input_data[0]
    print(payload)
    print(type(payload))
    inp_model_arr = input_data[1]
    for m in inp_model_arr:
        if m in model.keys():
            res = model[m].predict(xgb_encoders.csv_to_dmatrix(payload))
        output.append(res)
    return output
