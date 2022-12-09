import os
from multiprocessing import Process
import csv
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel

DIR = '/home/nicolas/hpc-home/ssod/'

def run_clip(images, image_names, dataset, data_type, save_path):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    # model.to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    inputs = processor(images=images, return_tensors="pt")
    # inputs.to(device)

    outputs = model(**inputs)
    # last_hidden_state = outputs.last_hidden_state
    pooled_output = outputs.pooler_output  # pooled CLS states

    print('Got Embeddings', flush=True)

    # open the file in the write mode
    f = open(f'{DIR}cluster_priors/clip_embeddings/{dataset}_{data_type}_clip.csv', 'a')
    # create the csv writer
    writer = csv.writer(f)
    # write a row to the csv file
    for i in range(len(pooled_output)):
        clip_embedding = list(pooled_output[i].cpu().detach().numpy())
        clip_embedding.insert(0, image_names[i])
        writer.writerow(clip_embedding)
    # close the file
    f.close()

def parse_image(image_fp):
    temp = Image.open(image_fp)
    image = temp.copy()
    temp.close()
    return image

def save_clip_embeddings(dataset, data_type, batch_size=64):
    # init
    data = f'{DIR}dataset/{dataset}/{data_type}/JPEGImages/'
    count = 0
    images = []
    image_names = []

    # open the file in the write mode
    f = open(f'{DIR}cluster_priors/clip_embeddings/{dataset}_{data_type}_clip.csv', 'w')
    f.close()

    # for each image
    for image_name in os.listdir(data):
        # skip pngs
        if image_name[-3:]=='png':
            continue
        # add image to lists
        count+=1
        image_fp = os.path.join(data, image_name)
        images.append(parse_image(image_fp))
        image_names.append(image_name)
        # generate clip embeddings based on batch size
        if count%batch_size == 0:
            print(count)
            # run process
            p = Process(target=run_clip, args=(images, image_names, dataset, data_type, DIR + f'cluster_priors/clip_embeddings/{dataset}_{data_type}_clip.csv'))
            p.start()
            p.join()
            # reset images
            images = []
            image_names = []
    
    # run process
    if images:
        p = Process(target=run_clip, args=(images, image_names, dataset, data_type, DIR + f'cluster_priors/clip_embeddings/{dataset}_{data_type}_clip_saved.csv'))
        p.start()
        p.join()


##############  GENERATE CLIP EMBEDDINGS #######################

save_clip_embeddings('C2F', 'unlabeled_data')