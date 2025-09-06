from r2egym.agenthub.run.edit import prepull_docker_images
import pandas as pd 
import glob
import argparse
from tqdm import tqdm

def get_docker_images(dataset_name):
    verl_parquet_path = glob.glob(dataset_name + "*_verl.parquet")[0]
    df = pd.read_parquet(verl_parquet_path)
    docker_images = set()
    for i in range(len(df)):
        try: 
            image = df["extra_info"][i]['docker_image']
        except KeyError:
            print(f"No docker image found for index {i} in dataset {dataset_name}")
            continue
        docker_images.add(image)
    return docker_images

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepull Docker images from datasets')
    parser.add_argument('--base_path', type=str, required=True, help='Base path to datasets')
    args = parser.parse_args()
    base_path = args.base_path
    
    datasets = [
        "d1",
        "d1_d2_mix",
        "d1_d3",
        "d1_d4",
        "SWE_Bench_Verified"
    ]

    total_set = set()
    for dataset in tqdm(datasets):
        dataset_path = base_path + dataset + "/"
        docker_images = get_docker_images(dataset_path)
        total_set.update(docker_images)


    prepull_docker_images(list(total_set), max_workers=100, timeout=600)