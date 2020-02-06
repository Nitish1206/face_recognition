import argparse
from glob import glob
from generate_embedings import generate_encodings,Train,Merge_encodings
from datetime import datetime

# command line argument setup


# dataset = args.dataset
def call_train(dataset,name):
    model_name = "aligned_test_320_240_512_700"
    # model_name="2020-01-27851720"
    file_name = model_name

    # generate face_encodings from encodings function
    obj1 = generate_encodings()
    file_name = obj1.encodings(dataset,file_name,name)

    # merge all single pickle files
    # print(file_name)
    # exit()
    obj2 = Merge_encodings()
    encodings_files = glob("single_pickle/*.pickle")
    # print(encodings_files)
    # exit()
    merge_file = obj2.merge(files=encodings_files)
    # merge_file="7_labled_merge"
    # train final model here.
    obj3 = Train()
    model_name_return = obj3.start_train(model_name,merge_file)

    print("Model-->",model_name_return,"Merge File Name-->",merge_file)

if __name__=="__main__":
    text = """
            -d --> Path to dataset
            -m --> Model name
            -f --> Encoding File name
            -mrf --> Merged Encodings File name
            """
    parser = argparse.ArgumentParser(description=text)
    parser.add_argument("--dataset", "-d", help="Path to your dataset", type=str, required=True)
    # parser.add_argument("--model_name", "-m", help="name of your model,label-file & Architecture", type=str, required=True)
    # parser.add_argument("--file_name", "-f", help="name of your encoded file", type=str, required=True)
    args = parser.parse_args()
    dataset = args.dataset
    call_train(dataset=dataset)